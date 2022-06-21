
# %%
def shortestPalindrome(s):
    s = list(s)
    sl = len(s)
    rptr = sl  # right-pointer

    while rptr != 0:
        subarr = s[:rptr]  # we check to see if this subarray is a palindrome
        print(subarr)
        print(rptr)
        if rptr % 2 == 0:
            lefthalf, righthalf = subarr[:rptr//2], subarr[rptr//2:]

        else:
            lefthalf, righthalf = subarr[:rptr//2], subarr[rptr//2 + 1:]
        print(lefthalf, righthalf)
        if lefthalf == righthalf[::-1]:  # if the subarray is a palindrome, stop
            print('break')
            break
        rptr -= 1

    leftpart = []  # the characters we add to the front of it
    # construct the front-part
    while rptr != sl:
        leftpart.insert(0, s[rptr])
        print(leftpart)
        rptr += 1
    return "".join(leftpart + s)
# %%


def shortestPalindrome(s):
    ans = 0
    i_s = s[::-1]
    for i in range(len(s)):
        print(i)
        print(s, i_s)
        i_s = i_s[:-1]
        s = s[1:]
        if i_s == s:
            print('break;')
            break

        ans += 1

    return ans
# %%


def shortestPalindrome(s: str) -> str:
    lth, i, n = 0, 1, len(s)
    if n == 0:
        return s
    s1, lps = s + '#' + s[::-1], [0] * (2 * n)
    while i < 2 * n:
        if s1[lth] == s1[i]:
            lth += 1
            lps[i] = lth
            i += 1
        elif lth > 0:
            lth = lps[lth - 1]
        else:
            i += 1
    return s[-1:lps[2 * n - 1]:-1] + s


# %%
s = 'yxssrmifb'
print('\nans', shortestPalindrome(s))

# %%


# def sub_lists(l):
#     lists = [[]]
#     for i in range(len(l) + 1):
#         for j in range(i):
#             lists.append(l[j: i])
#     return lists
def get_num_of_subset(n):
    print('get', n)
    ans = 0
    for i in range(1, n + 1):
        for j in range(i):
            ans += 1
    return ans


def bioHazard(n, allergic, poisonous):
    ans = n + get_num_of_subset(n)
    for i in range(len(allergic)):
        ans -= get_num_of_subset(n - abs(poisonous[i] - allergic[i]))
    return ans

# %%


# def get_subset(lst):
#     n = len(lst)
#     subset = []
#     for i in range(n + 1):
#         for j in range(i):
#             subset.append(lst[j: i])
#     return subset


# def bioHazard(n, allergic, poisonous):
#     ori_lst = list(range(1, n + 1))
#     subset = get_subset(ori_lst)


#     for i in range(len(allergic)):
#         lst= list(range(allergic[i], poisonous[i]+ 1))
#         print(lst)
#         print(get_subset(lst))
#         for sublst in get_subset(lst):
#             subset.remove(sublst)
#     print(subset)


# 1@5
# 1
# @
# 5
# 1@
# @5
# 1@5


# 234
# 234
# 1234
# 2345

# 23
# 123
# 234
# 12345
# 2345

# %%
print('ans', bioHazard(8, [2, 3, 4, 3], [8, 5, 6, 4]))
# %%
