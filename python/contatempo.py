# This function returns a string with the time elapsed between starttime,endtime


def tempo(starttime, endtime):

    dif = round(endtime - starttime, 2)
    d = int(dif / 86400)
    leftseconds = dif - d * 86400
    h = int(leftseconds / 3600)
    m = int((leftseconds / 3600 - h) * 60)
    s = round(((leftseconds / 3600 - h) * 60 - m) * 60, 2)

    if d > 0:
        string = (
            " The program ran for "
            + str(d)
            + " days, "
            + str(h)
            + " h, "
            + str(m)
            + " m"
        )
    elif h > 0:
        string = (
            " The program ran for "
            + str(h)
            + " h, "
            + str(m)
            + " m, "
            + str(int(s))
            + " s"
        )
    elif m > 0:
        string = " The program ran for " + str(m) + " m, " + str(s) + " s"
    else:
        string = " The program ran for " + str(dif) + " s"

    return string
