"""
dataset/generate_dataset.py
----------------------------
Generates a training_dataset.csv file with context-question pairs
inspired by SQuAD and SciQ formatting. This is used to train the
sentence-importance classifier model.
"""

import csv
import os

# ─────────────────────────────────────────────────────────────────
# Curated context-question pairs covering CS / networking topics
# Label: 1 = question-worthy sentence, 0 = background / filler
# ─────────────────────────────────────────────────────────────────
DATA = [
    # ── Networking ────────────────────────────────────────────────
    ("TCP/IP is the fundamental communication protocol of the Internet.",
     "What is TCP/IP?", 1),
    ("The OSI model divides network communication into seven distinct layers.",
     "How many layers does the OSI model have?", 1),
    ("DNS translates human-readable domain names into IP addresses.",
     "What is the role of DNS?", 1),
    ("HTTP is a stateless, application-layer protocol used to transfer hypertext.",
     "What is HTTP?", 1),
    ("A router forwards packets between different networks using routing tables.",
     "What is the function of a router?", 1),
    ("A switch operates at the data-link layer and forwards frames within a LAN.",
     "What does a switch do?", 1),
    ("HTTPS uses TLS/SSL to encrypt data between client and server.",
     "How does HTTPS secure communication?", 1),
    ("ARP maps an IP address to a physical MAC address on a local network.",
     "What is the purpose of ARP?", 1),
    ("UDP is a connectionless protocol that offers faster but unreliable delivery.",
     "What are the characteristics of UDP?", 1),
    ("The three-way handshake establishes a TCP connection: SYN, SYN-ACK, ACK.",
     "Describe the TCP three-way handshake.", 1),
    ("DHCP automatically assigns IP addresses to devices on a network.",
     "What does DHCP do?", 1),
    ("A firewall monitors and controls incoming and outgoing network traffic.",
     "What is a firewall?", 1),
    ("ICMP is used by network devices to send error messages and operational info.",
     "What is ICMP used for?", 1),
    ("Network topology describes the arrangement of nodes in a network.",
     "What is network topology?", 1),
    ("The transmission medium can be wired (Ethernet) or wireless (Wi-Fi).",
     "What types of transmission media exist?", 1),

    # ── Algorithms & Data Structures ──────────────────────────────
    ("Binary search finds an element in a sorted array in O(log n) time.",
     "What is the time complexity of binary search?", 1),
    ("A stack follows the Last-In-First-Out (LIFO) principle.",
     "What principle does a stack follow?", 1),
    ("A queue follows the First-In-First-Out (FIFO) principle.",
     "What principle does a queue follow?", 1),
    ("Merge sort divides the array recursively and merges sorted halves.",
     "How does merge sort work?", 1),
    ("Quick sort uses a pivot element and partitioning to sort in O(n log n).",
     "Explain the quick sort algorithm.", 1),
    ("Dijkstra's algorithm finds the shortest path from a source to all vertices.",
     "What does Dijkstra's algorithm compute?", 1),
    ("A binary tree is a tree where each node has at most two children.",
     "Define a binary tree.", 1),
    ("Dynamic programming solves problems by storing results of subproblems.",
     "What is dynamic programming?", 1),
    ("Bubble sort repeatedly swaps adjacent elements that are out of order.",
     "How does bubble sort work?", 1),
    ("Hashing maps data to fixed-size values for fast lookup.",
     "What is hashing?", 1),
    ("A linked list stores elements in nodes with pointers to the next node.",
     "What is a linked list?", 1),
    ("Breadth-first search explores nodes level by level using a queue.",
     "Describe BFS traversal.", 1),
    ("Depth-first search explores as far as possible along each branch.",
     "Describe DFS traversal.", 1),
    ("A heap is a complete binary tree that satisfies the heap property.",
     "What is a heap data structure?", 1),
    ("Recursion is a technique where a function calls itself to solve subproblems.",
     "What is recursion?", 1),

    # ── Operating Systems ─────────────────────────────────────────
    ("The operating system manages hardware resources and provides services.",
     "What is an operating system?", 1),
    ("A process is an instance of a program in execution.",
     "What is a process?", 1),
    ("Virtual memory allows a computer to use more memory than physically installed.",
     "What is virtual memory?", 1),
    ("Deadlock occurs when processes wait forever for resources held by each other.",
     "What is deadlock?", 1),
    ("A semaphore is a synchronization primitive used to control access.",
     "What is a semaphore?", 1),
    ("Paging divides memory into fixed-size pages for efficient management.",
     "What is paging in OS?", 1),
    ("The CPU scheduler decides which process runs next on the processor.",
     "What does the CPU scheduler do?", 1),
    ("Multithreading allows concurrent execution of threads within a process.",
     "What is multithreading?", 1),
    ("File systems organize and store files on storage devices.",
     "What is a file system?", 1),
    ("Interrupts signal the CPU that an event requires immediate attention.",
     "What are interrupts?", 1),

    # ── Database Systems ──────────────────────────────────────────
    ("SQL is a standard language for managing relational databases.",
     "What is SQL?", 1),
    ("Normalization reduces data redundancy and improves database integrity.",
     "What is normalization?", 1),
    ("ACID properties ensure reliable database transactions.",
     "What are ACID properties?", 1),
    ("An index speeds up database query performance.",
     "What is a database index?", 1),
    ("A primary key uniquely identifies each record in a table.",
     "What is a primary key?", 1),
    ("A join combines rows from two or more tables based on a related column.",
     "What is a SQL join?", 1),
    ("A foreign key establishes a link between two database tables.",
     "What is a foreign key?", 1),
    ("Stored procedures are precompiled SQL statements stored in the database.",
     "What are stored procedures?", 1),

    # ── Machine Learning ──────────────────────────────────────────
    ("Supervised learning trains a model on labelled input-output pairs.",
     "What is supervised learning?", 1),
    ("A neural network is a computational model inspired by the human brain.",
     "What is a neural network?", 1),
    ("Overfitting occurs when a model learns noise instead of patterns.",
     "What is overfitting?", 1),
    ("Cross-validation evaluates model performance on unseen data.",
     "What is cross-validation?", 1),
    ("Gradient descent minimizes a loss function by iteratively updating weights.",
     "What is gradient descent?", 1),
    ("A decision tree splits data into branches based on feature values.",
     "What is a decision tree?", 1),
    ("Clustering groups similar data points without labelled data.",
     "What is clustering?", 1),
    ("Regularization prevents overfitting by penalising model complexity.",
     "What is regularization?", 1),
    ("Feature engineering transforms raw data into meaningful inputs for models.",
     "What is feature engineering?", 1),
    ("Precision measures the fraction of true positives among predicted positives.",
     "What is precision in classification?", 1),

    # ── Filler / non-question-worthy sentences (label = 0) ────────
    ("This chapter introduces the topic discussed in later sections.", "", 0),
    ("Refer to the appendix for additional tables and figures.", "", 0),
    ("The following examples illustrate the concepts above.", "", 0),
    ("In summary, we have covered the main points.", "", 0),
    ("See figure 3.1 for a diagram of the architecture.", "", 0),
    ("Further reading is available in the references section.", "", 0),
    ("The next chapter will build upon these foundations.", "", 0),
    ("As mentioned earlier, we will revisit this point shortly.", "", 0),
    ("Please note the important terms highlighted in bold.", "", 0),
    ("The exercise at the end of this chapter tests your understanding.", "", 0),
]


def create_dataset(output_path: str = "dataset/training_dataset.csv") -> None:
    """Write the curated pairs to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["context", "question", "label"])
        for ctx, qst, lbl in DATA:
            writer.writerow([ctx, qst, lbl])
    print(f"[Dataset] Saved {len(DATA)} rows → {output_path}")


if __name__ == "__main__":
    create_dataset()
