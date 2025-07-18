#include <freertos/FreeRTOS.h>
#include <freertos/task.h>
#include <stdio.h>
#include <math.h>

// 模拟传感器数据结构体
typedef struct {
    float accel_x, accel_y, accel_z; // 加速度 (m/s^2)
    float gyro_x, gyro_y, gyro_z;   // 角速度 (rad/s)
    float latitude, longitude;       // GPS 位置 (度)
} SensorData;

// 控制输出结构体
typedef struct {
    float rudder_angle; // 舵面角度 (度)
    float thrust;       // 推进器推力 (0-1)
} ControlOutput;

// 卡尔曼滤波状态结构体
typedef struct {
    float x;        // 估计值
    float P;        // 估计误差协方差
    float Q;        // 过程噪声协方差
    float R;        // 测量噪声协方差
    float K;        // 卡尔曼增益
} KalmanFilter;

// 全局变量
SensorData sensor_data;
ControlOutput control_output;
KalmanFilter kf_position; // 用于位置估计
float target_lat = 40.0, target_lon = 116.5; // 目标位置

// 初始化卡尔曼滤波器
void initKalmanFilter(KalmanFilter *kf) {
    kf->x = 0.0;    // 初始估计值
    kf->P = 1.0;    // 初始协方差
    kf->Q = 0.01;   // 过程噪声
    kf->R = 0.1;    // 测量噪声
    kf->K = 0.0;    // 初始增益
}

// 卡尔曼滤波更新
float kalmanUpdate(KalmanFilter *kf, float measurement) {
    // 预测
    kf->P = kf->P + kf->Q;

    // 更新
    kf->K = kf->P / (kf->P + kf->R);
    kf->x = kf->x + kf->K * (measurement - kf->x);
    kf->P = (1 - kf->K) * kf->P;

    return kf->x;
}

// 数据采集任务
void dataAcquisitionTask(void *pvParameters) {
    while (1) {
        // 模拟传感器数据（实际中通过 ADC 或 I2C 读取）
        sensor_data.accel_x = 0.1 + (float)(rand() % 100) / 1000.0; // 模拟噪声
        sensor_data.gyro_y = 0.05 + (float)(rand() % 50) / 1000.0;
        sensor_data.latitude = 39.9042 + (float)(rand() % 100) / 10000.0;
        sensor_data.longitude = 116.4074 + (float)(rand() % 100) / 10000.0;

        printf("Raw Data: accel_x=%.3f, lat=%.4f, lon=%.4f\n",
               sensor_data.accel_x, sensor_data.latitude, sensor_data.longitude);
        vTaskDelay(pdMS_TO_TICKS(100)); // 每 100ms 采集一次
    }
}

// 导航与控制任务
void navigationTask(void *pvParameters) {
    // PID 参数
    float Kp = 1.0, Ki = 0.1, Kd = 0.05;
    float error_lat, error_lon, integral_lat = 0, integral_lon = 0;
    float derivative_lat, derivative_lon, last_error_lat = 0, last_error_lon = 0;
    float estimated_lat, estimated_lon;

    // 初始化卡尔曼滤波器
    initKalmanFilter(&kf_position);

    while (1) {
        // 使用卡尔曼滤波估计位置
        estimated_lat = kalmanUpdate(&kf_position, sensor_data.latitude);
        estimated_lon = kalmanUpdate(&kf_position, sensor_data.longitude);

        // 计算误差
        error_lat = target_lat - estimated_lat;
        error_lon = target_lon - estimated_lon;

        // PID 控制 - 纬度
        integral_lat += error_lat;
        derivative_lat = error_lat - last_error_lat;
        float correction_lat = Kp * error_lat + Ki * integral_lat + Kd * derivative_lat;

        // PID 控制 - 经度
        integral_lon += error_lon;
        derivative_lon = error_lon - last_error_lon;
        float correction_lon = Kp * error_lon + Ki * integral_lon + Kd * derivative_lon;

        // 控制输出
        control_output.rudder_angle = correction_lat + correction_lon; // 简化为单一舵面控制
        control_output.thrust = 0.8 + (fabs(error_lat) + fabs(error_lon)) * 0.1; // 动态调整推力
        if (control_output.thrust > 1.0) control_output.thrust = 1.0;

        // 打印控制结果
        printf("Control: rudder_angle=%.2f, thrust=%.2f, est_lat=%.4f, est_lon=%.4f\n",
               control_output.rudder_angle, control_output.thrust, estimated_lat, estimated_lon);

        // 更新误差
        last_error_lat = error_lat;
        last_error_lon = error_lon;

        vTaskDelay(pdMS_TO_TICKS(50)); // 每 50ms 计算一次
    }
}

// 故障检测任务
void faultDetectionTask(void *pvParameters) {
    while (1) {
        // 简单故障检测：检查传感器数据是否异常
        if (fabs(sensor_data.accel_x) > 10.0 || sensor_data.latitude < -90 || sensor_data.latitude > 90) {
            printf("Fault detected! Sensor data out of range.\n");
            // 应急处理：切换到备用模式或停止推进
            control_output.thrust = 0.0;
        }
        vTaskDelay(pdMS_TO_TICKS(500)); // 每 500ms 检查一次
    }
}

// 主函数
void app_main() {
    // 初始化控制输出
    control_output.rudder_angle = 0.0;
    control_output.thrust = 0.0;

    // 创建任务
    xTaskCreate(dataAcquisitionTask, "DataTask", 2048, NULL, 1, NULL);
    xTaskCreate(navigationTask, "NavTask", 2048, NULL, 2, NULL);
    xTaskCreate(faultDetectionTask, "FaultTask", 2048, NULL, 1, NULL);

    // 启动调度器
    vTaskStartScheduler();
}

// 模拟嵌入式环境的主函数入口
int main() {
    app_main();
    return 0;
}