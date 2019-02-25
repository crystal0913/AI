class Stack:
    """模拟栈"""

    def __init__(self):
        self.items = []

    def isEmpty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def top(self):
        if not self.isEmpty():
            return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)



# 栈顶为最大元素的序号
def find_right_big(a):
    res = [0] * len(a)
    sta = Stack()
    sta.push(0)
    for i in range(1, len(a)):
        while not sta.isEmpty() and a[i] > a[sta.top()]:
            res[sta.top()] = a[i]
            sta.pop()
        sta.push(i)
    return res


def quickSortOneTime(array, low, high):  # 一趟快速排序
    key = array[low]
    while low < high:
        while key < array[high] and low < high:
            high -= 1
        array[low] = array[high]
        while key > array[low] and low < high:
            low += 1
        array[high] = array[low]
        print(array)
    array[high] = key
    return high  # <class 'list'>: [13, 21, 5, 19, 37, 44, 80, 75, 88, 64, 92, 56]


def twoDepart(array, low, high, key):
    if low <= high:
        mid = quickSortOneTime(array, low, high)
    if array[mid] == key:
        return mid
    elif array[mid] < key:
        low = mid + 1
        return twoDepart(array, low, high, key)
    elif array[mid] > key:
        high = mid - 1
        return twoDepart(array, low, high, key)
    return -1


def quicksort(a):
    key = a[0]
    i = 0
    j = len(a) - 1
    while i<j:
        print(a)
        while key < a[j] and i < j:
            j -= 1
        if i<j:
            a[i]=a[j]
            i+=1
            while a[i] < key and i < j:
                i += 1
            if i<j:
                a[j]=a[i]
                j -= 1

    a[i] = key
    print(a)


def count(x):
    num = 0
    while x:
        print(x)
        num += 1
        x = x&(x-1)
    return num


class Solution:
    def binarySearch(self, nums, target):
        start = 0
        end = len(nums)
        while start < end:
            mid = (end + start) // 2
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                start = mid + 1
            else:
                end = mid - 1
        return -1

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        l = self.binarySearch(nums, target)
        if l == -1:
            return [-1, -1]
        r = l
        for i in range(l + 1, len(nums)):
            if nums[i] == target:
                r += 1
            else:
                break
        return [l, r]


def combinationSum(candidates, target):
    res = []
    if len(candidates) == 0:
        return res
    candidates.sort()
    dfs2(candidates, target, 0, [], res)
    return res


def dfs(nums, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        res.append(path)
        return
    for i in range(index, len(nums)):
        dfs(nums, target - nums[i], i, path + [nums[i]], res)


def dfs2(nums, target, index, path, res):
    if target < 0:
        return  # backtracking
    if target == 0:
        if path not in res:
            res.append(path)
        return
    for i in range(index, len(nums)):
        dfs2(nums, target - nums[i], i+1, path + [nums[i]], res)


if __name__ == "__main__":
    print(combinationSum([10,1,2,7,6,1,5], 8))
    # s = Solution()
    # print(s.searchRange([2,2], 2))
    exit(0)
    quicksort([46,30,82,90,56,17,95,15])
    # print(find_right_big([2, 7, 11, 5, 3, 1, 6, 8]))
    x = [44, 92, 5, 88, 13, 80, 19, 75, 21, 64, 37, 56]

    # print(twoDepart(x, 0, len(x)-1, 13))
    # print(x)
