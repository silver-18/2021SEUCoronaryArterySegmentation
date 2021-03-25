import numpy as np
import torch
import time

def reshape_img(image, z, y, x):
    out = np.zeros([z, y, x], dtype=np.float32)
    out[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]] \
        = image[0:image.shape[0], 0:image.shape[1], 0:image.shape[2]]
    return out


def read_file_from_txt(txt_path):
    files = []
    for line in open(txt_path, 'r'):
        files.append(line.strip())
    # print(files)
    return files


def predict(model, original_shape, image, num_classes):
    print("Predict test data")
    time_start = time.time()

    model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    # shape put into Unet
    shape = (224, 288, 288)

    x = original_shape[2]
    y = original_shape[1]
    z = original_shape[0]

    image = image.reshape(z, y, x)
    image = image.astype(np.float32)

    o = z
    p = y
    q = x

    if shape[0] > z:
        z = shape[0]
        image = reshape_img(image, z, y, x)
    if shape[1] > y:
        y = shape[1]
        image = reshape_img(image, z, y, x)
    if shape[2] > x:
        x = shape[2]
        image = reshape_img(image, z, y, x)

    predict = np.zeros([1, num_classes, z, y, x], dtype=np.float32)
    n_map = np.zeros([1, num_classes, z, y, x], dtype=np.float32)

    a = np.zeros(shape=shape)
    a = np.where(a == 0)
    map_kernal = 1 / ((a[0] - shape[0] // 2) ** 4 + (a[1] - shape[1] // 2) ** 4 + (a[2] - shape[2] // 2) ** 4 + 1)
    map_kernal = np.reshape(map_kernal, newshape=(1, 1,) + shape)

    image = image[np.newaxis, np.newaxis, :, :, :]
    stride_x = shape[0] // 2
    stride_y = shape[1] // 2
    stride_z = shape[2] // 2
    for i in range(z // stride_x - 1):
        for j in range(y // stride_y - 1):
            for k in range(x // stride_z - 1):
                image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                          k * stride_z:k * stride_z + shape[2]]
                image_i = torch.from_numpy(image_i)
                if torch.cuda.is_available():
                    image_i = image_i.cuda()
                output = model(image_i)
                output = output.data.cpu().numpy()

                predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += output * map_kernal

                n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                k * stride_z:k * stride_z + shape[2]] += map_kernal

            image_i = image[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
                      x - shape[2]:x]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()
            predict[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += output * map_kernal

            n_map[:, :, i * stride_x:i * stride_x + shape[0], j * stride_y:j * stride_y + shape[1],
            x - shape[2]:x] += map_kernal

        for k in range(x // stride_z - 1):
            image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
                      k * stride_z:k * stride_z + shape[2]]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()
            predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += output * map_kernal

            n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y,
            k * stride_z:k * stride_z + shape[2]] += map_kernal

        image_i = image[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += output * map_kernal
        n_map[:, :, i * stride_x:i * stride_x + shape[0], y - shape[1]:y, x - shape[2]:x] += map_kernal

    for j in range(y // stride_y - 1):
        for k in range((x - shape[2]) // stride_z):
            image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                      k * stride_z:k * stride_z + shape[2]]
            image_i = torch.from_numpy(image_i)
            if torch.cuda.is_available():
                image_i = image_i.cuda()
            output = model(image_i)
            output = output.data.cpu().numpy()

            predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            k * stride_z:k * stride_z + shape[2]] += output * map_kernal

            n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
            k * stride_z:k * stride_z + shape[2]] += map_kernal

        image_i = image[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
                  x - shape[2]:x]
        image_i = torch.from_numpy(image_i)  # 把数组转化成张量
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
        x - shape[2]:x] += output * map_kernal

        n_map[:, :, z - shape[0]:z, j * stride_y:j * stride_y + shape[1],
        x - shape[2]:x] += map_kernal

    for k in range(x // stride_z - 1):
        image_i = image[:, :, z - shape[0]:z, y - shape[1]:y,
                  k * stride_z:k * stride_z + shape[2]]
        image_i = torch.from_numpy(image_i)
        if torch.cuda.is_available():
            image_i = image_i.cuda()
        output = model(image_i)
        output = output.data.cpu().numpy()

        predict[:, :, z - shape[0]:z, y - shape[1]:y,
        k * stride_z:k * stride_z + shape[2]] += output * map_kernal

        n_map[:, :, z - shape[0]:z, y - shape[1]:y,
        k * stride_z:k * stride_z + shape[2]] += map_kernal

    image_i = image[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x]
    image_i = torch.from_numpy(image_i)
    if torch.cuda.is_available():
        image_i = image_i.cuda()
    output = model(image_i)
    output = output.data.cpu().numpy()

    predict[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += output * map_kernal
    n_map[:, :, z - shape[0]:z, y - shape[1]:y, x - shape[2]:x] += map_kernal

    predict = predict / n_map
    result = predict[0, 0, 0:o, 0:p, 0:q]

    # 0-1 normalization
    result = np.where(result > 0.5, 1, 0)

    time_end = time.time()
    print("Prediction Done in " + str(round(time_end - time_start, 2)) + " sec")
    return result
