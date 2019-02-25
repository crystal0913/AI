from stack import Stack


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left, self.right = None, None

    def __str__(self):
        return str(self.val)


class Solution:
    def __init__(self):
        self.balance = True

    def arrayConvertBst(self, lists, start, end):
        if start > end:
            return None
        mid = (start + end) // 2
        cur = TreeNode(lists[mid])
        cur.left = self.arrayConvertBst(lists, start, mid-1)
        cur.right = self.arrayConvertBst(lists, mid+1, end)
        return cur

    def isBalanceTree(self, tree):
        self.getDepth(tree)
        return self.balance

    def getDepth(self, tree):
        if not tree or self.balance == False:
            return 0
        left = self.getDepth(tree.left)
        right = self.getDepth(tree.right)
        if abs(left - right) > 1:
            self.balance = False
        return left + 1 if left>right else right + 1

    def preOrderTraverasl(self, tree):
        node = tree
        stacks = Stack()
        res = []
        while node or not stacks.isEmpty():
            if node:
                res.append(node.val)
                stacks.push(node)
                node = node.left
            else:
                top = stacks.pop()
                node = top.right
        return res

    def inOrderTraverasl(self, tree):
        node = tree
        stacks = Stack()
        res = []
        while node or not stacks.isEmpty():
            if node:
                stacks.push(node)
                node = node.left
            else:
                top = stacks.pop()
                res.append(top.val)
                node = top.right
        return res

    def postOrderTraverasl(self, tree):
        stacks = Stack()
        res = []
        stacks.push(tree)
        cur = pre = None
        while not stacks.isEmpty():
            cur = stacks.top()
            if not (cur.left or cur.right) or (pre and (pre == cur.left or pre == cur.right)):
                res.append(cur.val)
                stacks.pop()
                pre = cur
            else:
                if cur.right:
                    stacks.push(cur.right)
                if cur.left:
                    stacks.push(cur.left)
        return res

    def isBSTree(self, tree):
        node = tree
        stacks = Stack()
        pre = None
        while node or not stacks.isEmpty():
            if node:
                stacks.push(node)
                node = node.left
            else:
                top = stacks.pop()
                if pre and pre > top.val:
                    return False
                pre = top.val
                node = top.right
        return True

    def getPath(self, tree, target):
        st = Stack()
        action_stack = Stack()
        self._getPath(tree, target, st, action_stack, True)
        while not st.isEmpty():
            print(st.pop(), action_stack.pop())

    def _getPath(self, tree, target, stacks, action_stack, left):
        if not tree:
            return False
        stacks.push(tree)
        if left:
            action_stack.push(0)
        else:
            action_stack.push(1)
        if tree.val == target:
            return True
        b = False
        if tree.left:
            b = self._getPath(tree.left, target, stacks, action_stack, True)
        if not b and tree.right:
            b = self._getPath(tree.right, target, stacks, action_stack, False)
        if not b:
            stacks.pop()
            action_stack.pop()
        return b

    def lowestCommonAncestor(self, root, p, q):
        if not root or p == root.val or q == root.val:
            return root
        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)
        if left and right:
            return root
        if left:
            return left
        return right

    def printAllPath(self, tree, path, pathlen):
        if not tree:
            return
        path[pathlen] = tree.val
        pathlen += 1
        if not tree.left and not tree.right:
            print(path)
        else:
            self.printAllPath(tree.left, path, pathlen)
            self.printAllPath(tree.right, path, pathlen)


Tree = TreeNode(2)
bb = Tree.left = TreeNode(1)
bb.left = TreeNode(6)
bb.right = TreeNode(7)
bb.right.left = TreeNode(8)
aa=Tree.right = TreeNode(3)
aa.right=TreeNode(4)
aa.right.right=TreeNode(5)
s = Solution()

# s.getPath(Tree, 5)
x = [None for i in range(10)]
s.printAllPath(Tree, x, 0)

# print(s.lowestCommonAncestor(Tree, 7, 6))

