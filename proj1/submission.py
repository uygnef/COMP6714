## Submission.py for COMP6714-Project1

###################################################################################################################
## Question No. 0:
def add(a, b): # do not change the heading of the function
    return a + b




###################################################################################################################
## Question No. 1:

def gallop_to(a, val):# do not change the heading of the function
    
    count = 0
    delta = 1
    while(True):
        count += 1
        if not a.elem():
            break   
        if a.elem() < val:
            a.cur += delta
            delta *= 2
            continue
        elif a.elem() > val:
            a.cur = a.cur - delta//4
        break      
    return count


###################################################################################################################
## Question No. 2:

def Logarithmic_merge(index, cut_off, buffer_size): # do not change the heading of the function
    disk = [[]]
    memory = []

    def merge(i, sub_list, disk):
        if len(disk) == i + 1:
            disk.append([])
        disk[i+1].append(sorted(sub_list[0] + sub_list[1]))
        disk[i] = []
        if len(disk[i+1]) > 1:
            merge(i+1, disk[i+1], disk)
            
    def merge_disk(disk):
        if len(disk[0]) > 1:
            merge(0, disk[0], disk)
    
    
    for i in index[:cut_off]:
        memory.append(i)

        if len(memory) == 3:
            disk[0].append(sorted(memory))
            merge_disk(disk)
            memory = []
    
    disk = [[memory]] + disk
   # merge_disk(disk)
    ret = []
    for i in disk:
        ret.append(i[0])
    return ret            



###################################################################################################################
## Question No. 3:

def decode_gamma(inputs):# do not change the heading of the function
    all_digits = []
    while(inputs):
        k_d, remain = inputs.split("0", 1)
        k_r = remain[:len(k_d)]
        inputs = remain[len(k_d):]
        all_digits.append(2**len(k_d) + int(k_r, 2))   
    return all_digits

def decode_delta(inputs):# do not change the heading of the function
    import math
    all_digits = []
    while inputs:
        k_dd, remain = inputs.split("0", 1)
        k_dr = remain[:len(k_dd)]
        k_r_remain = remain[len(k_dd):]
        
        k_d = 2**(2**len(k_dd)) + int(k_dr, 2) - 1
        k_r = k_r_remain[:int(math.log2(k_d))]

        inputs = k_r_remain[int(math.log2(k_d)):]
        all_digits.append(k_d + int(k_r, 2))
    
    return all_digits

def decode_rice(inputs, b):# do not change the heading of the function
    import math
    all_digits = []
    b = int(math.log2(b))
    while(inputs):
        q, remain = inputs.split("0", 1)
        offset = remain[:b]
        inputs = remain[b:]
        all_digits.append(2**(len(q)) + int(offset, 2))
    return all_digits# **replace** this line with your code