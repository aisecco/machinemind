def generate(total, class_total, x, y):
    _image = []
    _label = []

    for i in range(0, total):
        for i0 in range(0, class_total):
            _image_col = []
            for i1 in range(0, y):
                _image_row = []
                for i2 in range(0, x):
                    v2 = i0 / 5
                    _image_row.append(v2)
                _image_col.append(_image_row)
            _image.append(_image_col)
            _label.append(i0)
    return (_image, _label)