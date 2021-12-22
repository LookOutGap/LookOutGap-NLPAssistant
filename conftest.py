import pandas as pd


def sents_chars_expected():
    return pd.Series(data=["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "abcdefghijklmnopqrstuvwxyz",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           ""
                           ])


def sents_nums_expected():
    return pd.Series(data=["",
                           "",
                           "0123456789",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           "",
                           ""
                           ])


def sents_all_expected():
    return pd.Series(data=["ABCDEFGHIJKLMNOPQRSTUVWXYZ",
                           "abcdefghijklmnopqrstuvwxyz",
                           "0123456789",
                           "",
                           ",./;'[]\-=",
                           '<>?:"{}|_+',
                           '!@#$%^&*()`~"'