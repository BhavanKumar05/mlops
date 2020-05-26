import smtplib
from email.mime.multipart import MIMEMultipart 
from email.mime.text import MIMEText


host_address ="bhavankumar05071998@gmail.com"
host_pass = "hemlatagangwar"
guest_address ="bhavankumar0507@gmail.com"
subject = "Regarding failure of lenetprogram.py"
content = '''hello, your lenetprogram code is incorrect '''
message = MIMEMultipart()
message['From'] = host_address
message['To'] = guest_address
message['Subject'] = subject
message.attach(MIMEText(content, 'plain'))
session = smtplib.SMTP('smtp.gmail.com',587)
session.starttls()
session.login(host_address,host_pass)
text = message.as_string()
session.sendmail(host_address,guest_address,text)
session.quit()
print("successfull")

 