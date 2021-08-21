import cv2,glob,os


imgs = glob.glob('test/*.jpg')
saved_dir = 'preprocess'
os.makedirs(saved_dir,exist_ok=True)
for path in imgs:
    name = os.path.basename(path)
    src = cv2.imread(path)
    width = src.shape[1]
    steps = round(width/768.0)
    new_width = steps*768
    src = cv2.resize(src, (new_width, int(new_width/1.5)))
    print(src.shape)
    for i in range(steps):
        for j in range(steps):
            x0 = i*768
            x1 = (i+1)*768
            y0 = j*512
            y1 = (j+1)*512
            img = src[y0:y1, x0:x1, :]
            print(x0,x1,y0, y1,img.shape)
            cv2.imwrite(os.path.join(saved_dir, name+'_%d_%d.png'%(x0,y0)), img)