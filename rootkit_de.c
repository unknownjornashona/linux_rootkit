// einfaches LKM (Loadable Kernel Module) - Nur für Bildungszwecke
#include <linux/module.h>
#include <linux/kernel.h>

/* Modul-Initialisierung */
static int __init mein_module_init(void) {
    printk(KERN_INFO "Modul geladen: Hallo Kernel!\n");  // Kernel-Log schreiben
    return 0;  // Erfolg
}

/* Modul-Entfernung */
static void __exit mein_module_exit(void) {
    printk(KERN_INFO "Modul entfernt: Auf Wiedersehen!\n");
}

module_init(mein_module_init);  // Initialisierungsfunktion registrieren
module_exit(mein_module_exit); // Aufräumfunktion registrieren

MODULE_LICENSE("GPL");          // Lizenz (Pflicht!)
MODULE_AUTHOR("Dein Name");     // Autor
MODULE_DESCRIPTION("Ein pädagogisches LKM-Beispiel"); 