import cv2
import numpy as np
import base64
from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from ultralytics import YOLO
from PIL import Image

model = YOLO('best.pt')

@csrf_exempt
def detect_waldo(request):
    if request.method == 'POST':
        print("button is working properly")
        img_data = request.FILES.get('image')

        # Convert uploaded image to OpenCV format
        img_array = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = img.shape[:2]
        print(f"Image resolution: {width}x{height}")

        # Tiling parameters
        tile_size = 160
        overlap = 10

        # Loop over tiles
        for y in range(0, height, tile_size - overlap):
            for x in range(0, width, tile_size - overlap):
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                tile = img[y:y_end, x:x_end]

                # Run model on tile
                results = model(tile)[0]

                # Draw results on original image (adjusted coords)
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    global_x1 = x1 + x
                    global_y1 = y1 + y
                    global_x2 = x2 + x
                    global_y2 = y2 + y

                    # Blue rectangle
                    cv2.rectangle(img, (global_x1, global_y1), (global_x2, global_y2), (255, 0, 0), 2)

        # Encode and return image as base64
        _, buffer = cv2.imencode('.png', img)
        img_base64 = base64.b64encode(buffer).decode('utf-8')

        return JsonResponse({'processed_image': 'data:image/png;base64,' + img_base64})

    return render(request, 'main/main.html')

@csrf_exempt
def upload_and_detect(request):
    processed_img_b64 = None
    print("here we go")

    if request.method == 'POST' and request.FILES.get('image'):
        print("this function has been called properly")
        img_data = request.FILES['image']
        img_array = np.asarray(bytearray(img_data.read()), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        height, width = img.shape[:2]
        tile_size = 64
        overlap = 5  # You can use it later if needed

        all_predictions = []

        # Run detection tile by tile
        for y in range(0, height, tile_size):
            for x in range(0, width, tile_size):
                x_end = min(x + tile_size, width)
                y_end = min(y + tile_size, height)
                tile = img[y:y_end, x:x_end]

                results = model(tile)[0]

                for box in results.boxes:
                    conf = float(box.conf[0])
                    if conf > 0.3:  # Confidence threshold
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        global_x1 = x1 + x
                        global_y1 = y1 + y
                        global_x2 = x2 + x
                        global_y2 = y2 + y
                        all_predictions.append((conf, (global_x1, global_y1, global_x2, global_y2)))

        # Sort all predictions by confidence (descending)
        all_predictions.sort(reverse=True, key=lambda tup: tup[0])

        # Take top 1 to 3 predictions
        top_predictions = all_predictions[:3]

        # Draw selected predictions (with wider boxes)
        if top_predictions:
            for conf, (x1, y1, x2, y2) in top_predictions:
                # Expand the box by 10px on each side
                pad = 10
                x1 = max(x1 - pad, 0)
                y1 = max(y1 - pad, 0)
                x2 = min(x2 + pad, width - 1)
                y2 = min(y2 + pad, height - 1)

                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 5)
                print(f"Prediction with confidence {conf:.2f} at ({x1}, {y1}), ({x2}, {y2})")
        else:
            print("No confident predictions found.")

        # Encode the processed image
        _, buffer = cv2.imencode('.png', img)
        processed_img_b64 = base64.b64encode(buffer).decode('utf-8')

    return render(request, 'main/upload_detect.html', {
        'processed_image': processed_img_b64,
    })
