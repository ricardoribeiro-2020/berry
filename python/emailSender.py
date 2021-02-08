import smtplib

sender = 'ricardo.ribeiro@physics.org'
receiver = 'ricardo.ribeiro@physics.org'
smtp = 'localhost'
port0 = 25

message = """From: ricardo.ribeiro@physics.org
To: ricardo.ribeiro@physics.org
Subject: calculation

The calculation is finished.
"""

server = smtplib.SMTP(host=smtp,port=port0)
server.sendmail(sender, receiver, message)         
server.quit()
print("Successfully sent email")

