text = ["London Paris London","Paris Paris London"]
cur_word = None
count_1 = 0
count_2 = 0

for line in text:
    words = line.split()
    for word in words:
        if cur_word is None:
            cur_word = word
        if word == cur_word:
            count_1 += 1
        else:
            count_2 += 1
    print(f"{line} has count_1 = {count_1} and count_2 = {count_2}")
    count_1 = 0
    count_2 = 0
        


