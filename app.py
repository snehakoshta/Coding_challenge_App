import streamlit as st
import random



st.markdown("""
        <style>
  

            @keyframes gradientBG {
                0% {background-position: 0% 50%;}
                50% {background-position: 100% 50%;}
                100% {background-position: 0% 50%;}
            }
            
            .navbar {
                position: fixed;
                top: 25px;
                left: 10px;
                width: 200px;
                height: 95vh;
                background: gray;
                padding-top: 20px;
                display: flex;
                flex-direction: column;
                align-items: center;
                border-radius: 15px;
            }

            .stAppView {
                margin-left: 220px;
            }

            .navbar a {
                color: white;
                text-decoration: none;
                font-size: 18px;
                padding: 15px 20px;
                width: 100%;
                text-align: center;
                border-radius: 10px;
            }
            
            .navbar a:hover {
                background: darkgray;
            }
        </style>
     
    """, unsafe_allow_html=True)



dsa_problems = {
   

    "Google": {
        "Basic": [
        {
            "problem": "Reverse a string",
            "python": "def reverse_string(s): return s[::-1]",
            "java": "public String reverseString(String s) { return new StringBuilder(s).reverse().toString(); }",
            "c": "#include <stdio.h>\nvoid reverseString(char s[]) { int l = strlen(s); for(int i=0; i<l/2; i++) { char temp = s[i]; s[i] = s[l-i-1]; s[l-i-1] = temp; } }"
        },
        {
            "problem": "Find the factorial of a number",
            "python": "def factorial(n): return 1 if n==0 else n * factorial(n-1)",
            "java": "public int factorial(int n) { return (n == 0) ? 1 : n * factorial(n - 1); }",
            "c": "#include <stdio.h>\nint factorial(int n) { return (n == 0) ? 1 : n * factorial(n - 1); }"
        },
        {
            "problem": "Find the largest element in an array",
            "python": "def find_max(arr): return max(arr)",
            "java": "public int findMax(int[] arr) { return Arrays.stream(arr).max().getAsInt(); }",
            "c": "#include <stdio.h>\nint findMax(int arr[], int n) { int max = arr[0]; for(int i=1; i<n; i++) if(arr[i] > max) max = arr[i]; return max; }"
        }
    ],
        

        "Medium": [
            {
                "problem": "Find the first non-repeating character in a string",
                "python": 'def first_unique_char(s): return next((c for c in s if s.count(c) == 1), None)',
                "java": 'public char firstUniqueChar(String s) { for(char c : s.toCharArray()) if(s.indexOf(c) == s.lastIndexOf(c)) return c; return \' \'; }',
                "c": '#include <stdio.h>\nchar firstUniqueChar(char *s) { int count[256] = {0}; for(int i=0; s[i]; i++) count[s[i]]++; for(int i=0; s[i]; i++) if(count[s[i]] == 1) return s[i]; return \' \'; }'
            }
        ],
        "Hard": [
    {
        "problem": "Find the shortest path in a graph (Dijkstra's Algorithm)",
        "python": "# Implement Dijkstra’s algorithm for shortest path\nimport heapq\n\ndef dijkstra(graph, start):\n    heap = [(0, start)]\n    distances = {node: float('inf') for node in graph}\n    distances[start] = 0\n\n    while heap:\n        current_distance, current_node = heapq.heappop(heap)\n        if current_distance > distances[current_node]:\n            continue\n\n        for neighbor, weight in graph[current_node]:\n            distance = current_distance + weight\n            if distance < distances[neighbor]:\n                distances[neighbor] = distance\n                heapq.heappush(heap, (distance, neighbor))\n    return distances",
        
        "java": "// Implement Dijkstra’s algorithm in Java\nimport java.util.*;\n\nclass Graph {\n    public static Map<String, Integer> dijkstra(Map<String, List<int[]>> graph, String start) {\n        PriorityQueue<int[]> heap = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));\n        Map<String, Integer> distances = new HashMap<>();\n        for (String node : graph.keySet()) distances.put(node, Integer.MAX_VALUE);\n        distances.put(start, 0);\n        heap.add(new int[]{0, start.hashCode()});\n\n        while (!heap.isEmpty()) {\n            int[] current = heap.poll();\n            int currentDistance = current[0];\n            String currentNode = String.valueOf(current[1]);\n\n            for (int[] neighbor : graph.get(currentNode)) {\n                int newDist = currentDistance + neighbor[1];\n                if (newDist < distances.get(String.valueOf(neighbor[0]))) {\n                    distances.put(String.valueOf(neighbor[0]), newDist);\n                    heap.add(new int[]{newDist, neighbor[0]});\n                }\n            }\n        }\n        return distances;\n    }\n}",
        
        "c": "// Implement Dijkstra’s algorithm in C\n#include <stdio.h>\n#include <limits.h>\n#include <stdbool.h>\n#define V 9\n\nint minDistance(int dist[], bool sptSet[]) {\n    int min = INT_MAX, min_index;\n    for (int v = 0; v < V; v++) {\n        if (sptSet[v] == false && dist[v] <= min) {\n            min = dist[v], min_index = v;\n        }\n    }\n    return min_index;\n}\n\nvoid dijkstra(int graph[V][V], int src) {\n    int dist[V];\n    bool sptSet[V];\n    for (int i = 0; i < V; i++) dist[i] = INT_MAX, sptSet[i] = false;\n    dist[src] = 0;\n    for (int count = 0; count < V - 1; count++) {\n        int u = minDistance(dist, sptSet);\n        sptSet[u] = true;\n        for (int v = 0; v < V; v++) {\n            if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) {\n                dist[v] = dist[u] + graph[u][v];\n            }\n        }\n    }\n    for (int i = 0; i < V; i++) {\n        printf(\"Vertex %d -> Distance %d\\n\", i, dist[i]);\n    }\n}"
    }
]
    },
  
    
    "Microsoft": {
        "Basic": [
            {
                "problem": "Find the second largest element in an array",
                "python": 'def second_largest(arr): return sorted(arr)[-2]',
                "java": 'public int secondLargest(int[] arr) { Arrays.sort(arr); return arr[arr.length - 2]; }',
                "c": '#include <stdio.h>\nint secondLargest(int arr[], int n) { int first = arr[0], second = -1; for(int i=1; i<n; i++) if(arr[i] > first) { second = first; first = arr[i]; } return second; }'
            }
        ]
    },
    

"TCS": {
    "Basic": [
        {
            "problem": "Check if a number is prime",
            "python": "def is_prime(n): return all(n%i!=0 for i in range(2, int(n**0.5)+1))",
            "java": "public boolean isPrime(int n) { for(int i = 2; i <= Math.sqrt(n); i++) if(n % i == 0) return false; return true; }",
            "c": "#include <stdio.h>\nint isPrime(int n) { for(int i = 2; i*i <= n; i++) if(n % i == 0) return 0; return 1; }"
        },
        {
            "problem": "Reverse a string",
            "python": "def reverse_string(s): return s[::-1]",
            "java": "public String reverseString(String s) { return new StringBuilder(s).reverse().toString(); }",
            "c": "#include <stdio.h>\nvoid reverseString(char s[]) { int l = strlen(s); for(int i=0; i<l/2; i++) { char temp = s[i]; s[i] = s[l-i-1]; s[l-i-1] = temp; } }"
        },
        {
            "problem": "Find the factorial of a number",
            "python": "def factorial(n): return 1 if n==0 else n * factorial(n-1)",
            "java": "public int factorial(int n) { return (n == 0) ? 1 : n * factorial(n - 1); }",
            "c": "#include <stdio.h>\nint factorial(int n) { return (n == 0) ? 1 : n * factorial(n - 1); }"
        },
        {
            "problem": "Check if a string is a palindrome",
            "python": "def is_palindrome(s): return s == s[::-1]",
            "java": "public boolean isPalindrome(String s) { return new StringBuilder(s).reverse().toString().equals(s); }",
            "c": "#include <stdio.h>\nint isPalindrome(char s[]) { int l = strlen(s); for(int i=0; i<l/2; i++) if(s[i] != s[l-i-1]) return 0; return 1; }"
        },
        {
            "problem": "Find the greatest common divisor (GCD) of two numbers",
            "python": "import math\ndef gcd(a, b): return math.gcd(a, b)",
            "java": "public int gcd(int a, int b) { return (b == 0) ? a : gcd(b, a % b); }",
            "c": "#include <stdio.h>\nint gcd(int a, int b) { return (b == 0) ? a : gcd(b, a % b); }"
        },
        {
            "problem": "Print Fibonacci series up to n terms",
            "python": "def fibonacci(n): return [0, 1] + [sum([0, 1][-2:]) for _ in range(n-2)]",
            "java": "public void fibonacci(int n) { int a = 0, b = 1; for(int i=0; i<n; i++) { System.out.print(a + \" \"); int temp = a; a = b; b = temp + b; } }",
            "c": "#include <stdio.h>\nvoid fibonacci(int n) { int a = 0, b = 1; for(int i=0; i<n; i++) { printf(\"%d \", a); int temp = a; a = b; b = temp + b; } }"
        }
    ],


    "Medium": [
        {
            "problem": "Find the first non-repeating character in a string",
            "python": "def first_unique_char(s): return next((c for c in s if s.count(c) == 1), None)",
            "java": "public char firstUniqueChar(String s) { for(char c : s.toCharArray()) if(s.indexOf(c) == s.lastIndexOf(c)) return c; return ' '; }",
            "c": "#include <stdio.h>\nchar firstUniqueChar(char *s) { int count[256] = {0}; for(int i=0; s[i]; i++) count[s[i]]++; for(int i=0; s[i]; i++) if(count[s[i]] == 1) return s[i]; return ' '; }"
        },
        {
            "problem": "Push all zeros to the end of an array",
            "python": "def move_zeros(arr): return [num for num in arr if num != 0] + [0] * arr.count(0)",
            "java": "public int[] moveZeros(int[] arr) { return Arrays.stream(arr).filter(num -> num != 0).toArray(); }",
            "c": "#include <stdio.h>\nvoid moveZeros(int arr[], int n) { int count = 0; for(int i=0; i<n; i++) if(arr[i] != 0) arr[count++] = arr[i]; while(count < n) arr[count++] = 0; }"
        },
        {
            "problem": "Find the missing number in an array",
            "python": "def find_missing_number(arr, n): return sum(range(1, n+1)) - sum(arr)",
            "java": "public int findMissingNumber(int[] arr, int n) { return IntStream.rangeClosed(1, n).sum() - Arrays.stream(arr).sum(); }",
            "c": "#include <stdio.h>\nint findMissingNumber(int arr[], int n) { int sum = (n * (n + 1)) / 2; for(int i = 0; i < n-1; i++) sum -= arr[i]; return sum; }"
        }
     ],
    "Hard": [
        {
            "problem": "Implement Dijkstra’s Algorithm",
            "python": "import heapq\ndef dijkstra(graph, start): heap = [(0, start)]; distances = {node: float('inf') for node in graph}; distances[start] = 0; while heap: current_distance, current_node = heapq.heappop(heap); if current_distance > distances[current_node]: continue; for neighbor, weight in graph[current_node].items(): distance = current_distance + weight; if distance < distances[neighbor]: distances[neighbor] = distance; heapq.heappush(heap, (distance, neighbor)); return distances",
            "java": "import java.util.*; class Graph { Map<Integer, Map<Integer, Integer>> graph = new HashMap<>(); public Map<Integer, Integer> dijkstra(int start) { PriorityQueue<int[]> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a[0])); Map<Integer, Integer> distances = new HashMap<>(); graph.keySet().forEach(node -> distances.put(node, Integer.MAX_VALUE)); distances.put(start, 0); pq.add(new int[]{0, start}); while (!pq.isEmpty()) { int[] current = pq.poll(); int currentDistance = current[0], currentNode = current[1]; if (currentDistance > distances.get(currentNode)) continue; for (Map.Entry<Integer, Integer> neighbor : graph.get(currentNode).entrySet()) { int distance = currentDistance + neighbor.getValue(); if (distance < distances.get(neighbor.getKey())) { distances.put(neighbor.getKey(), distance); pq.add(new int[]{distance, neighbor.getKey()}); } } } return distances; } }",
            "c": "#include <stdio.h>\n#include <limits.h>\n#define V 5\nint minDistance(int dist[], int sptSet[]) { int min = INT_MAX, min_index; for (int v = 0; v < V; v++) if (sptSet[v] == 0 && dist[v] <= min) min = dist[v], min_index = v; return min_index; }\nvoid dijkstra(int graph[V][V], int src) { int dist[V]; int sptSet[V]; for (int i = 0; i < V; i++) dist[i] = INT_MAX, sptSet[i] = 0; dist[src] = 0; for (int count = 0; count < V - 1; count++) { int u = minDistance(dist, sptSet); sptSet[u] = 1; for (int v = 0; v < V; v++) if (!sptSet[v] && graph[u][v] && dist[u] != INT_MAX && dist[u] + graph[u][v] < dist[v]) dist[v] = dist[u] + graph[u][v]; } }"
        },
        {
            "problem": "Solve the N-Queens problem",
            "python": "def solve_n_queens(n): def is_safe(board, row, col): for i in range(row): if board[i] == col or abs(board[i] - col) == row - i: return False; return True; def solve(board, row): if row == n: solutions.append(board[:]); return; for col in range(n): if is_safe(board, row, col): board[row] = col; solve(board, row + 1); solutions = []; solve([-1] * n, 0); return solutions",
            "java": "class NQueens { static boolean isSafe(int board[], int row, int col) { for (int i = 0; i < row; i++) if (board[i] == col || Math.abs(board[i] - col) == row - i) return false; return true; } static void solve(int board[], int row, int n, List<int[]> solutions) { if (row == n) { solutions.add(board.clone()); return; } for (int col = 0; col < n; col++) if (isSafe(board, row, col)) { board[row] = col; solve(board, row + 1, n, solutions); } } static List<int[]> solveNQueens(int n) { List<int[]> solutions = new ArrayList<>(); solve(new int[n], 0, n, solutions); return solutions; } }",
            "c": "#include <stdio.h>\n#define N 8\nint board[N]; int isSafe(int row, int col) { for (int i = 0; i < row; i++) if (board[i] == col || abs(board[i] - col) == row - i) return 0; return 1; } void solve(int row) { if (row == N) { for (int i = 0; i < N; i++) printf(\"%d \", board[i]); printf(\"\\n\"); return; } for (int col = 0; col < N; col++) if (isSafe(row, col)) { board[row] = col; solve(row + 1); } } int main() { solve(0); return 0; }"
        }
    ],


}
}
dsa_problems["Amazon"] = {
    "Basic": [
        {
            "problem": "Check if two strings are anagrams",
            "python": "def is_anagram(s1, s2): return sorted(s1) == sorted(s2)",
            "java": "public boolean isAnagram(String s1, String s2) { return Arrays.equals(s1.chars().sorted().toArray(), s2.chars().sorted().toArray()); }",
            "c": "#include <stdio.h>\nint isAnagram(char *s1, char *s2) { int count[256] = {0}; while (*s1) count[*s1++]++; while (*s2) count[*s2--]--; for(int i=0;i<256;i++) if(count[i]) return 0; return 1; }"
        },
        {
            "problem": "Find the longest common prefix",
            "python": "def longest_common_prefix(strs): return ''.join(c[0] for c in zip(*strs) if len(set(c)) == 1)",
            "java": "public String longestCommonPrefix(String[] strs) { Arrays.sort(strs); String first = strs[0], last = strs[strs.length - 1]; int i = 0; while (i < first.length() && first.charAt(i) == last.charAt(i)) i++; return first.substring(0, i); }",
            "c": "#include <stdio.h>\nchar* longestCommonPrefix(char *strs[], int n) { if (n == 0) return \"\"; char *prefix = strs[0]; for (int i=1; i<n; i++) { int j=0; while (prefix[j] && strs[i][j] && prefix[j] == strs[i][j]) j++; prefix[j] = '\\0'; } return prefix; }"
        }
    ],
    "Medium": [
        {
            "problem": "Implement a stack using linked list",
            "python": '''class Node:\n    def __init__(self, data): self.data = data; self.next = None\nclass Stack:\n    def __init__(self): self.top = None\n    def push(self, x): new_node = Node(x); new_node.next = self.top; self.top = new_node\n    def pop(self): if self.top is None: return None; temp = self.top.data; self.top = self.top.next; return temp\n    def peek(self): return self.top.data if self.top else None''',
            "java": '''class Node { int data; Node next; public Node(int data) { this.data = data; this.next = null; }}\nclass Stack {\n    private Node top;\n    public void push(int x) { Node newNode = new Node(x); newNode.next = top; top = newNode; }\n    public int pop() { if (top == null) return -1; int temp = top.data; top = top.next; return temp; }\n    public int peek() { return (top != null) ? top.data : -1; }\n}''',
            "c": '''#include <stdio.h>\n#include <stdlib.h>\ntypedef struct Node { int data; struct Node* next; } Node;\nNode* top = NULL;\nvoid push(int x) { Node* newNode = (Node*)malloc(sizeof(Node)); newNode->data = x; newNode->next = top; top = newNode; }\nint pop() { if (top == NULL) return -1; int temp = top->data; top = top->next; return temp; }\nint peek() { return (top != NULL) ? top->data : -1; }'''
        }
    ],
    "Hard": [
        {
            "problem": "Find the lowest common ancestor in a binary tree",
            "python": '''class TreeNode:\n    def __init__(self, val=0, left=None, right=None): self.val = val; self.left = left; self.right = right\n    def lca(root, p, q): if not root or root == p or root == q: return root\n    left = lca(root.left, p, q); right = lca(root.right, p, q)\n    return root if left and right else left or right''',
            "java": '''class TreeNode { int val; TreeNode left, right; TreeNode(int x) { val = x; }}\nclass Solution {\n    public TreeNode lca(TreeNode root, TreeNode p, TreeNode q) { if (root == null || root == p || root == q) return root; TreeNode left = lca(root.left, p, q); TreeNode right = lca(root.right, p, q); return left != null && right != null ? root : (left != null ? left : right); }\n}''',
            "c": '''#include <stdio.h>\ntypedef struct TreeNode { int val; struct TreeNode* left, *right; } TreeNode;\nTreeNode* lca(TreeNode* root, TreeNode* p, TreeNode* q) { if (!root || root == p || root == q) return root; TreeNode* left = lca(root->left, p, q); TreeNode* right = lca(root->right, p, q); return left && right ? root : left ? left : right; }'''
        }
    ]
}


dsa_problems["Infosys"] = {
    "Basic": [
        {
            "problem": "Implement Fibonacci sequence",
            "python": "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
            "java": "public int fibonacci(int n) { return (n <= 1) ? n : fibonacci(n - 1) + fibonacci(n - 2); }",
            "c": "#include <stdio.h>\nint fibonacci(int n) { return (n <= 1) ? n : fibonacci(n - 1) + fibonacci(n - 2); }"
        },
        {
            "problem": "Merge two sorted arrays",
            "python": "def merge_sorted(arr1, arr2): return sorted(arr1 + arr2)",
            "java": "public int[] mergeSortedArrays(int[] arr1, int[] arr2) { return IntStream.concat(Arrays.stream(arr1), Arrays.stream(arr2)).sorted().toArray(); }",
            "c": "#include <stdio.h>\nvoid mergeSortedArrays(int arr1[], int arr2[], int n1, int n2, int merged[]) { int i=0, j=0, k=0; while (i<n1 && j<n2) merged[k++] = (arr1[i] < arr2[j]) ? arr1[i++] : arr2[j++]; while (i<n1) merged[k++] = arr1[i++]; while (j<n2) merged[k++] = arr2[j++]; }"
        }
    ]
}
dsa_problems["Infosys"]["Medium"] = [
    {
        "problem": "Find the intersection of two arrays",
        "python": "def intersection(arr1, arr2): return list(set(arr1) & set(arr2))",
        "java": "public int[] intersection(int[] arr1, int[] arr2) { return Arrays.stream(arr1).filter(num -> Arrays.stream(arr2).anyMatch(n -> n == num)).toArray(); }",
        "c": "#include <stdio.h>\nvoid intersection(int arr1[], int n1, int arr2[], int n2) { for (int i=0; i<n1; i++) for (int j=0; j<n2; j++) if (arr1[i] == arr2[j]) printf(\"%d \", arr1[i]); }"
    },
    {
        "problem": "Find the majority element in an array",
        "python": "from collections import Counter\ndef majority_element(arr): return max(Counter(arr).items(), key=lambda x: x[1])[0]",
        "java": "public int majorityElement(int[] arr) { Map<Integer, Integer> countMap = new HashMap<>(); for(int num : arr) countMap.put(num, countMap.getOrDefault(num, 0) + 1); return countMap.entrySet().stream().max(Map.Entry.comparingByValue()).get().getKey(); }",
        "c": "#include <stdio.h>\nint majorityElement(int arr[], int n) { int count[256] = {0}; for(int i=0; i<n; i++) count[arr[i]]++; int max_count = 0, res = -1; for(int i=0; i<256; i++) if(count[i] > max_count) max_count = count[i], res = i; return res; }"
    },
    {
        "problem": "Find the missing number in a sequence",
        "python": "def find_missing(arr, n): return sum(range(1, n+1)) - sum(arr)",
        "java": "public int findMissing(int[] arr, int n) { return IntStream.rangeClosed(1, n).sum() - Arrays.stream(arr).sum(); }",
        "c": "#include <stdio.h>\nint findMissing(int arr[], int n) { int sum = (n * (n + 1)) / 2; for (int i = 0; i < n - 1; i++) sum -= arr[i]; return sum; }"
    },
    {
        "problem": "Push all zeros to the end of an array",
        "python": "def move_zeros(arr): return [num for num in arr if num != 0] + [0] * arr.count(0)",
        "java": "public int[] moveZeros(int[] arr) { return Arrays.stream(arr).filter(num -> num != 0).toArray(); }",
        "c": "#include <stdio.h>\nvoid moveZeros(int arr[], int n) { int count = 0; for(int i=0; i<n; i++) if(arr[i] != 0) arr[count++] = arr[i]; while(count < n) arr[count++] = 0; }"
    }
]
dsa_problems["Accenture"] = {
    "Basic": [
        {
            "problem": "Find the height of a binary tree",
            "python": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.left = None\n        self.right = None\n\ndef tree_height(root):\n    if root is None:\n        return 0\n    return 1 + max(tree_height(root.left), tree_height(root.right))",
            "java": "class Node {\n    int data;\n    Node left, right;\n    Node(int item) { data = item; left = right = null; }\n}\n\nclass Tree {\n    int treeHeight(Node root) {\n        if (root == null) return 0;\n        return 1 + Math.max(treeHeight(root.left), treeHeight(root.right));\n    }\n}",
            "c": "#include <stdio.h>\n#include <stdlib.h>\n\nstruct Node {\n    int data;\n    struct Node* left;\n    struct Node* right;\n};\n\nint treeHeight(struct Node* root) {\n    if (root == NULL) return 0;\n    return 1 + ((treeHeight(root->left) > treeHeight(root->right)) ? treeHeight(root->left) : treeHeight(root->right));\n}"
        }
    ],
    "Medium": [
        {
            "problem": "Maximum Path Sum in a Binary Tree",
            "python": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.left = None\n        self.right = None\n\ndef max_path_sum(root):\n    def helper(node):\n        if not node:\n            return 0\n        left = max(helper(node.left), 0)\n        right = max(helper(node.right), 0)\n        max_sum[0] = max(max_sum[0], node.data + left + right)\n        return node.data + max(left, right)\n    max_sum = [float('-inf')]\n    helper(root)\n    return max_sum[0]",
            "java": "class Node {\n    int data;\n    Node left, right;\n    Node(int item) { data = item; left = right = null; }\n}\n\nclass Tree {\n    int maxSum = Integer.MIN_VALUE;\n    int maxPathSum(Node root) {\n        helper(root);\n        return maxSum;\n    }\n    int helper(Node node) {\n        if (node == null) return 0;\n        int left = Math.max(helper(node.left), 0);\n        int right = Math.max(helper(node.right), 0);\n        maxSum = Math.max(maxSum, node.data + left + right);\n        return node.data + Math.max(left, right);\n    }\n}",
            "c": "#include <stdio.h>\n#include <limits.h>\n\nstruct Node {\n    int data;\n    struct Node* left;\n    struct Node* right;\n};\n\nint maxSum = INT_MIN;\nint maxPathSumHelper(struct Node* root) {\n    if (!root) return 0;\n    int left = maxPathSumHelper(root->left);\n    int right = maxPathSumHelper(root->right);\n    maxSum = (maxSum > root->data + left + right) ? maxSum : root->data + left + right;\n    return root->data + (left > right ? left : right);\n}\nint maxPathSum(struct Node* root) {\n    maxPathSumHelper(root);\n    return maxSum;\n}"
        }
    ],
    "Hard": [
        {
            "problem": "Diameter of a Binary Tree using DP",
            "python": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.left = None\n        self.right = None\n\ndef diameter(root):\n    def helper(node):\n        if not node:\n            return 0\n        left = helper(node.left)\n        right = helper(node.right)\n        max_diameter[0] = max(max_diameter[0], left + right)\n        return max(left, right) + 1\n    max_diameter = [0]\n    helper(root)\n    return max_diameter[0]",
            "java": "class Node {\n    int data;\n    Node left, right;\n    Node(int item) { data = item; left = right = null; }\n}\n\nclass Tree {\n    int maxDiameter = 0;\n    int diameter(Node root) {\n        helper(root);\n        return maxDiameter;\n    }\n    int helper(Node node) {\n        if (node == null) return 0;\n        int left = helper(node.left);\n        int right = helper(node.right);\n        maxDiameter = Math.max(maxDiameter, left + right);\n        return Math.max(left, right) + 1;\n    }\n}",
            "c": "#include <stdio.h>\n#include <stdlib.h>\n\nstruct Node {\n    int data;\n    struct Node* left;\n    struct Node* right;\n};\n\nint maxDiameter = 0;\nint diameterHelper(struct Node* root) {\n    if (!root) return 0;\n    int left = diameterHelper(root->left);\n    int right = diameterHelper(root->right);\n    maxDiameter = (maxDiameter > left + right) ? maxDiameter : left + right;\n    return (left > right ? left : right) + 1;\n}\nint diameter(struct Node* root) {\n    diameterHelper(root);\n    return maxDiameter;\n}"
        }
    ]
}
dsa_problems["Deloitte"] = {
    "Basic": [
        {
            "problem": "Check if a Binary Tree is Symmetric",
            "python": "class Node:\n    def __init__(self, data):\n        self.data = data\n        self.left = None\n        self.right = None\n\ndef isSymmetric(root):\n    def isMirror(t1, t2):\n        if not t1 and not t2:\n            return True\n        if not t1 or not t2:\n            return False\n        return (t1.data == t2.data and isMirror(t1.left, t2.right) and isMirror(t1.right, t2.left))\n    return isMirror(root, root)",
            "java": "class Node {\n    int data;\n    Node left, right;\n    Node(int item) { data = item; left = right = null; }\n}\n\nclass Tree {\n    boolean isSymmetric(Node root) {\n        return isMirror(root, root);\n    }\n\n    boolean isMirror(Node t1, Node t2) {\n        if (t1 == null && t2 == null) return true;\n        if (t1 == null || t2 == null) return false;\n        return (t1.data == t2.data) && isMirror(t1.left, t2.right) && isMirror(t1.right, t2.left);\n    }\n}",
            "c": "#include <stdio.h>\n#include <stdbool.h>\n\nstruct Node {\n    int data;\n    struct Node* left;\n    struct Node* right;\n};\n\nbool isMirror(struct Node* t1, struct Node* t2) {\n    if (!t1 && !t2) return true;\n    if (!t1 || !t2) return false;\n    return (t1->data == t2->data) && isMirror(t1->left, t2->right) && isMirror(t1->right, t2->left);\n}\n\nbool isSymmetric(struct Node* root) {\n    return isMirror(root, root);\n}"
        }
    ]
}


# Ensure every company has all difficulty levels
for company in dsa_problems.keys():
    for level in ["Basic", "Medium", "Hard"]:
        if level not in dsa_problems[company]:
            dsa_problems[company][level] = []  

# Custom CSS for Bigger Font in Challenge Box
st.markdown("""
    <style>
        .title-text {
            font-size: 40px;
            text-align: center;
            color: #FF5733;
            font-weight: bold;
        }
        .section-header {
            font-size: 28px;
            color: #4CAF50;
            font-weight: bold;
        }
        .problem-text {
            font-size: 22px;
            color: #D32F2F;
            padding: 10px;
            background-color: #f9f9f9;
            border-radius: 10px;
            border-left: 6px solid #FF5733;
            margin-bottom: 15px;
        }
        .custom-button {
            background-color: #FF5733; 
            color: white;
            border-radius: 10px;
            padding: 12px 20px;
            font-size: 18px;
            font-weight: bold;
            border: none;
            cursor: pointer;
            display: inline-block;
            width: 100%;
        }
        .custom-button:hover {
            background-color: #C70039;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit UI
st.markdown("<div class='title-text'>🚀 CodeQuest: DSA Challenge</div>", unsafe_allow_html=True)
st.write("Attempt a Data Structures & Algorithms challenge from top tech companies!")

# Iterate over companies to create sections
for company in dsa_problems.keys():
    with st.expander(f"🔹 {company} Challenges"):
        difficulty = st.selectbox(f"Choose Difficulty Level for {company}", ["Basic", "Medium", "Hard"], key=company)

        # Styled Button using CSS
        button_html = f"""<button class='custom-button' onclick="window.location.reload()">Generate {company} Challenge</button>"""
        st.markdown(button_html, unsafe_allow_html=True)

        if st.button(f"Generate {company} Challenge", key=company+"_btn"):  
            if dsa_problems[company][difficulty]:  
                problem_data = random.choice(dsa_problems[company][difficulty])

                st.markdown(f"<div class='section-header'>🧠 {company} DSA Coding Challenge:</div>", unsafe_allow_html=True)
                st.markdown(f"<div class='problem-text'><b>{problem_data['problem']}</b></div>", unsafe_allow_html=True)

                st.subheader("💻 Solutions in Python, Java, and C:")
                st.code(problem_data["python"], language="python")
                st.code(problem_data["java"], language="java")
                st.code(problem_data["c"], language="c")
            else:
                st.warning(f"No problems found for {company} at {difficulty} level. Try another level.")

        # User solution input
        user_solution = st.text_area(f"Write your solution here for {company}:", key=company+"_solution")

        if st.button(f"Submit Solution for {company}", key=company+"_submit"):
            if user_solution.strip():
                st.success("✅ Solution submitted! Keep refining your skills.")
            else:
                st.warning("⚠️ Please write a solution before submitting.")