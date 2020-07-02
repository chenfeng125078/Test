import os
import glob


if __name__ == '__main__':
    base_dir = os.path.join("D:\\data\\分类_晶点", "*.bmp")
    image_list = glob.glob(base_dir)
    print(len(image_list))
    i = 1
    for item in image_list:
        # print(item)
        old_image = item.split("\\")[-1]
        old_name = old_image.split(".")[0]
        # print(old_name)
        new_name = str(i) + "(3)"
        new_image = new_name + ".bmp"
        # print(new_image)
        new_path_list = item.split("\\")[:-1]
        # print(new_path_list)
        new_path_list.append(new_image)
        # print(new_path_list)
        new_path = "D:\\"
        for path in new_path_list[1:]:
            print(path)
            new_path = os.path.join(new_path, path)
        print(new_path)
        # print(new_path)
        # print(item, new_path)
        os.rename(item, new_path)
        i += 1

