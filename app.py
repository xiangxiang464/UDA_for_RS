import base64
import json
import os
import time
import math

from io import BytesIO

import cv2
from flask import Flask, request, render_template, send_file, jsonify
from werkzeug.utils import secure_filename

from tools.test import run_test
from flask_cors import CORS



app = Flask(__name__)
CORS(app)

@app.route('/')
def hello_world():
    return 'Hello World!'


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查请求中是否包含文件
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    # 如果用户没有选择文件，浏览器可能会提交一个没有文件名的空部分
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and allowed_file(file.filename):
        # 为了安全，使用Werkzeug提供的secure_filename方法
        filename = secure_filename(file.filename)
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        choice = request.form.get('num')
        # print(choice)
        config_dir = ["configs/uda_rs/potsdam2isprs_uda_pt7_dw_local7_label_warm_daformer_mitb5.py",
                      "configs/uda_rs/isprs2potsdam_uda_pt7_dw_local7_label_warm_daformer_mitb5.py"]
        checkpoint_dir = [
            "work_dirs/20240326_183914_potsdam2isprs_uda_pt7_dw_local7_label_warm_daformer_mitb5/iter_4000.pth",
            "work_dirs/20240326_183914_potsdam2isprs_uda_pt7_dw_local7_label_warm_daformer_mitb5/iter_4000.pth"]
        # 保存文件
        save_path = os.path.join('data/vaihingen/assets', timestamp, filename)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        file.save(save_path)
        show_dir = os.path.join('data/vaihingen/assets', timestamp, 'pred')
        eval_out = run_test(config_dir[int(choice)],
                 checkpoint_dir[int(choice)],
                 path=os.path.join("assets",timestamp),show_dir=show_dir)
        pred_img = os.path.join(show_dir, filename)
        image = cv2.imread(pred_img)
        _, buffer = cv2.imencode('.jpg', image)
        # 发送图片文件给前端
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # 从原始数据提取IoU和Acc值，并替换nan值为None
        iou_data = {k.split('.')[-1]: round(v, 4) if not (isinstance(v, float) and math.isnan(v)) else None for k, v in
                    eval_out.items() if k.startswith('IoU.')}
        acc_data = {k.split('.')[-1]: round(v, 4) if not (isinstance(v, float) and math.isnan(v)) else None for k, v in
                    eval_out.items() if k.startswith('Acc.')}

        # 结果列表
        organized_data = [iou_data, acc_data]

        print(organized_data)
        # 构建JSON响应
        response = {
            'message': 'Image loaded successfully',
            'eval': organized_data,
            'image_base64': encoded_image
        }

        # 返回JSON响应
        return jsonify(response), 200

    return jsonify({'error': 'File extension not allowed'}), 400


@app.route('/send_image')
def send_image():
    # 图片文件路径
    image_filename = 'area2_0_0_512_512.png'
    image_path = os.path.join('assets', image_filename)
    image = cv2.imread(image_path)
    _, buffer = cv2.imencode('.jpg', image)
    buffer = BytesIO(buffer.tobytes())
    # 发送图片文件给前端
    return send_file(buffer, mimetype='image/jpg')


if __name__ == '__main__':
    app.run()