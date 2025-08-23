#!/usr/bin/env python3
import requests
import time
import subprocess
import logging
import os
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header

# 配置日志
logging.basicConfig(
    filename='monitor.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 邮件配置
EMAIL_CONFIG = {
    'smtp_server': 'smtp.qq.com',    # QQ邮箱SMTP服务器
    'smtp_port': 465,                # QQ邮箱SSL端口
    'sender_email': '2530947719@qq.com',  # 你的QQ邮箱
    'sender_password': 'gdqvaahvgioiecij',  # QQ邮箱SMTP授权码
    'receiver_email': '2530947719@qq.com'  # 接收通知的邮箱
}

def send_email(subject, body):
    """发送邮件通知"""
    try:
        msg = MIMEMultipart()
        msg['From'] = Header(EMAIL_CONFIG['sender_email'])
        msg['To'] = Header(EMAIL_CONFIG['receiver_email'])
        msg['Subject'] = Header(subject)
        
        # 添加正文
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # 使用SSL连接SMTP服务器
        with smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            try:
                server.sendmail(
                    EMAIL_CONFIG['sender_email'],
                    EMAIL_CONFIG['receiver_email'],
                    msg.as_string()
                )
                logging.info(f"Email notification sent successfully: {subject}")
            except smtplib.SMTPException as e:
                # 检查是否是特定的无害错误
                if isinstance(e.args[0], tuple) and e.args[0][0] == -1 and e.args[0][1] == b'\x00\x00\x00':
                    logging.info(f"Email notification sent successfully (ignored known harmless error): {subject}")
                else:
                    raise  # 重新抛出其他类型的SMTP错误
    except Exception as e:
        logging.error(f"Failed to send email notification: {e}")

def kill_port_6006():
    """杀掉6006端口的进程"""
    try:
        cmd = "lsof -ti:6006"
        pid = subprocess.check_output(cmd, shell=True).decode().strip()
        if pid:
            subprocess.run(f"kill -9 {pid}", shell=True)
            logging.info("Successfully killed process on port 6006")
    except subprocess.CalledProcessError:
        logging.info("No process found on port 6006")

def clear_gpu_memory():
    """清理GPU显存"""
    try:
        subprocess.run("nvidia-smi --gpu-reset", shell=True)
        logging.info("Successfully cleared GPU memory")
    except Exception as e:
        logging.error(f"Error clearing GPU memory: {e}")

def restart_server():
    """重启服务器"""
    try:
        # 进入screen会话并启动服务
        cmd = """screen -S gunicorn -X stuff $'\n'"""  # 发送换行确保在新行
        subprocess.run(cmd, shell=True)
        
        cmd = """screen -S gunicorn -X stuff 'gunicorn -w 1 -b 0.0.0.0:6006 --timeout 180 server:app\n'"""
        subprocess.run(cmd, shell=True)
        
        logging.info("Server restart command sent")
    except Exception as e:
        logging.error(f"Error restarting server: {e}")

def check_health():
    """检查服务器健康状态"""
    try:
        response = requests.get('https://u310022-a1f3-69f8bbcd.nmb1.seetacloud.com:8443/health', timeout=10)
        return response.status_code == 200 and response.json()['status'] == 'success'
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return False

def monitor_loop():
    """主监控循环"""
    consecutive_failures = 0  # 连续失败次数
    while True:
        logging.info("Performing health check...")
        if not check_health():
            consecutive_failures += 1
            logging.warning(f"Health check failed (Attempt {consecutive_failures}), initiating recovery process")
            
            # 1. 杀掉6006端口
            kill_port_6006()
            
            # 2. 清理显存
            clear_gpu_memory()
            
            # 3. 等待几秒钟确保所有清理工作完成
            time.sleep(5)
            
            # 4. 重启服务器
            restart_server()
            
            # 5. 等待服务器启动
            time.sleep(10)
            
            # 6. 验证服务器是否成功重启
            if check_health():
                logging.info("Server successfully recovered")
                consecutive_failures = 0  # 重置失败计数
                
                # 发送恢复通知
                if consecutive_failures >= 2:  # 如果之前连续失败超过2次，发送恢复通知
                    send_email(
                        "服务器已恢复",
                        f"服务器在经过{consecutive_failures}次失败后已成功恢复。\n恢复时间：{datetime.now()}"
                    )
            else:
                logging.error("Server recovery failed")
                # 发送警告邮件（连续失败2次后）
                if consecutive_failures >= 2:
                    send_email(
                        "服务器故障警告",
                        f"服务器健康检查连续失败{consecutive_failures}次，请检查。\n最后检查时间：{datetime.now()}\n"
                        f"服务地址：https://u310022-a1f3-69f8bbcd.nmb1.seetacloud.com:8443/health"
                    )
        else:
            logging.info("Health check passed")
            consecutive_failures = 0  # 重置失败计数
        
        # 等待3分钟
        time.sleep(180)

if __name__ == "__main__":
    logging.info("Starting server monitor")
    try:
        # 启动时发送通知
        send_email(
            "服务器监控已启动",
            f"服务器监控程序已于 {datetime.now()} 启动，开始监控服务状态。"
        )
        monitor_loop()
    except KeyboardInterrupt:
        logging.info("Monitor stopped by user")
        send_email(
            "服务器监控已停止",
            f"服务器监控程序已于 {datetime.now()} 被用户手动停止。"
        )
    except Exception as e:
        logging.error(f"Monitor stopped due to error: {e}")
        send_email(
            "服务器监控异常退出",
            f"服务器监控程序遇到错误并退出：\n{str(e)}\n时间：{datetime.now()}"
        ) 