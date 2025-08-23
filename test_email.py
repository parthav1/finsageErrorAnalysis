#!/usr/bin/env python3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
import socket

# 从monitor_server.py导入配置
from monitor_server import EMAIL_CONFIG

def test_smtp_connection():
    print(f"正在测试连接 {EMAIL_CONFIG['smtp_server']}:{EMAIL_CONFIG['smtp_port']}...")
    
    try:
        # 测试网络连接
        socket.create_connection((EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']), timeout=10)
        print("网络连接测试成功！")
    except socket.timeout:
        print("错误：连接超时，可能是网络问题或防火墙限制")
        return
    except socket.error as e:
        print(f"错误：网络连接失败 - {str(e)}")
        return

    try:
        # 创建邮件
        msg = MIMEMultipart()
        msg['From'] = Header(EMAIL_CONFIG['sender_email'])
        msg['To'] = Header(EMAIL_CONFIG['receiver_email'])
        msg['Subject'] = Header("测试邮件 - 服务器监控")
        msg.attach(MIMEText("这是一封测试邮件，用于验证服务器监控的邮件发送功能是否正常工作。", 'plain', 'utf-8'))

        # 使用SSL连接SMTP服务器
        print("正在连接SMTP服务器...")
        with smtplib.SMTP_SSL(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port'], timeout=30) as server:
            print("正在登录...")
            server.login(EMAIL_CONFIG['sender_email'], EMAIL_CONFIG['sender_password'])
            print("正在发送邮件...")
            try:
                server.sendmail(
                    EMAIL_CONFIG['sender_email'],
                    EMAIL_CONFIG['receiver_email'],
                    msg.as_string()
                )
                print("测试邮件发送成功！")
            except smtplib.SMTPException as e:
                # 检查是否是特定的无害错误
                if isinstance(e.args[0], tuple) and e.args[0][0] == -1 and e.args[0][1] == b'\x00\x00\x00':
                    print("测试邮件发送成功！（忽略已知的无害错误）")
                else:
                    raise  # 重新抛出其他类型的SMTP错误
    except smtplib.SMTPAuthenticationError:
        print("错误：认证失败，请检查QQ邮箱地址和SMTP授权码是否正确")
    except smtplib.SMTPException as e:
        print(f"SMTP错误：{str(e)}")
    except Exception as e:
        print(f"发生未知错误：{str(e)}")

if __name__ == "__main__":
    test_smtp_connection() 