## Submission.py for COMP6714-Project1

###################################################################################################################
## Question No. 0:
def add(a, b): # do not change the heading of the function
    return a + b




###################################################################################################################
## Question No. 1:

def gallop_to(a, val):# do not change the heading of the function
    def binary_search(start, end, a, val):
        while start < end:
            mid = (start + end) // 2
            if a.peek( mid ) < val:
                start = max(start + 1, mid)
            elif a.peek(mid) > val:
                end = min(mid, end - 1)
            else:
                end = mid
                break
        a.cur = end
        return
    a.cur = 1  
    count = 0
    while(a.elem() and a.elem() < val):
        count += 1
        a.cur *= 2
    binary_search(a.cur // 2, min(a.cur, len(a.data)) , a, val)   

###################################################################################################################
## Question No. 2:

def Logarithmic_merge(index, cut_off, buffer_size): # do not change the function heading
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

        if len(memory) >= buffer_size:
            disk[0].append(sorted(memory))
            merge_disk(disk)
            memory = []
    
    disk = [[memory]] + disk
    ret = []
    for i in disk:
        if i:
            ret += i
        else:
            ret.append(i)
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

def decode_delta(inputs):# do not change the function heading
    if(inputs == "0"):
        return [1]
    import math
    all_digits = []
    while inputs:
        if(inputs == "0"):
            all_digits.append(1)
            if len(inputs) == 1:
                return all_digits
            else:
                inputs = inputs[1:]
                continue
        k_dd, remain = inputs.split("0", 1)
        k_dr = remain[:len(k_dd)]
        k_r_remain = remain[len(k_dd):]

        
        k_d = 2**(2**len(k_dd)-1+ int(k_dr, 2))
        k_r = k_r_remain[:int(math.log2(k_d))]

        inputs = k_r_remain[int(math.log2(k_d)):]
        all_digits.append(k_d + int(k_r, 2))
    
    return all_digits

def decode_rice(inputs, b):# do not change the function heading
    import math
    all_digits = []
    sb = int(math.log2(b))
    while(inputs):
        q, remain = inputs.split("0", 1)
        offset = remain[:sb]
        inputs = remain[sb:]
        all_digits.append(b * len(q) + int(offset, 2))
    return all_digits