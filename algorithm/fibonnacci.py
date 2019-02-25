
def fibonnacci1(n):
    if n == 1:
        return 0
    elif n == 2:
        return 1
    else:
        f1 = 0
        f2 = 1
        res = 0
        for i in range(3, n+1):
            res = f1 + f2
            f1 = f2
            f2 = res
    return res


def fibonnacci2(first, second, n):
    if n <= 2:
        return 1
    elif n==3:
        return first + second
    else:
        return fibonnacci2(second, first+second, n-1)


print(fibonnacci1(7))
print(fibonnacci2(0, 1, 7))