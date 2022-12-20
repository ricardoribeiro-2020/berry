"""
   Script to send an email warning the end of a calculation

   Should be edited to conform to the users specifications
"""

import smtplib
from socket import gaierror
import berry._subroutines.loadmeta as m

# pylint: disable=C0103
###################################################################################

sender = "sender@physics.org"
receiver = "receiver@physics.org"
smtpserver = "localhost"
port0 = 25
login = ""
password = ""

message = """\
From: ricardo.ribeiro@physics.org
To: ricardo.ribeiro@physics.org
Subject: Calculation  """ + m.refname + """

The calculation is finished in directory\n""" + m.workdir + """

"""

try:
    with smtplib.SMTP(host=smtpserver, port=port0) as server:
        if login != "":
            server.login(login, password)
        server.sendmail(sender, receiver, message)
        server.quit()

except (gaierror, ConnectionRefusedError):
    print(" Failed to connect to server.")
except smtplib.SMTPException as e:
    print("SMTP error occurred.")
else:
    print("Successfully sent email")
