# mailer.py
# Utilizes voyutils.mailinator
# to send updates on simulations.
from voyutils import mailinator
import time

def mail(argv, stime, seed_num, seed_total, ucsd_email=True):
    """
    Sends email updates to Simon's email on simulation
    progress

    Parameters
    ----------
    argv: sys.argv
    stime: float
    seed_num: int
    seed_total: int
    ucsd_email: bool (Optional)
    """
    if len(argv) < 2:
        raise ValueError("Pass 'decryptor' password as script arg for smtp code: see Voytek whiteboard")

    # Accounting for 0-indexing
    seed_num += 1
    timer = time.time() - stime

    top = ''.join([str(seed_num), '/', str(seed_total), ' MC simulations complete.'])
    time_str = ''.join(['Block execution time: ', '{:.3f}'.format(timer)])
    msg = "\n".join([top, time_str])

    decryptor = argv[1]

    if len(argv) > 2:
        pw_filename = argv[2]
    elif ucsd_email:
        pw_filename = '../data/random/ucsd_pw.simon'
    else:
        pw_filename = '../data/random/gmail_pw.simon'

    pw = mailinator.get_password(decryptor, pw_filename)

    if ucsd_email:
        sending_email = 'shaxby@ucsd.edu'
    else:
        sending_email = 'simon.haxby@gmail.com'

    msg = mailinator.format_email(msg, argv[0],
	    ('Simon', 'simon.haxby@gmail.com'), ('Simon', sending_email))

    if ucsd_email:
        mailinator.ucsdmail(msg, pw)
    else:
        mailinator.gmail(msg, pw)