#git add .
#git commit -m "Add changes comment here"
#git push origin main

import cv2
import numpy as np
import pygame
import time
import threading
from collections import deque

class MotionActivatedLighting:
    def __init__(self):
        # Webカメラの設定
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("カメラにアクセスできません")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # モーション検出のパラメータ
        self.motion_threshold = 5000  # モーション検出の閾値
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)  # 過去のモーション量を保存
        
        # ライティングの設定
        self.light_color = (255, 255, 255)  # デフォルトの色
        self.light_intensity = 0.5  # 0.0 - 1.0の範囲
        
        # PyGameの設定
        pygame.init()
        self.screen_width = 800
        self.screen_height = 600
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("モーションアクティベーテッドライティング")
        
        # 各領域のライトの設定（位置と半径）
        self.lights = [
            {"pos": (200, 150), "radius": 100, "color": (255, 255, 255), "intensity": 0.5},
            {"pos": (600, 150), "radius": 100, "color": (255, 255, 255), "intensity": 0.5},
            {"pos": (200, 450), "radius": 100, "color": (255, 255, 255), "intensity": 0.5},
            {"pos": (600, 450), "radius": 100, "color": (255, 255, 255), "intensity": 0.5},
            {"pos": (400, 300), "radius": 150, "color": (255, 255, 255), "intensity": 0.5},
        ]
        
        # トラッキングする主要な動きポイント
        self.motion_points = []
        
        # スレッド関連
        self.running = True
        self.camera_thread = threading.Thread(target=self.process_camera)
        self.render_thread = threading.Thread(target=self.render_lights)
    
    def start(self):
        """システムを開始する"""
        self.camera_thread.start()
        self.render_thread.start()
        
        print("モーションアクティベーテッドライティングシステムを開始しました")
        print("終了するには'q'キーを押してください")
        
        try:
            while self.running:
                time.sleep(0.1)
                # メインスレッドでのキー入力チェック
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_q:
                            self.running = False
        except KeyboardInterrupt:
            self.running = False
        
        self.camera_thread.join()
        self.render_thread.join()
        self.cleanup()
    
    def process_camera(self):
        """カメラからの映像を処理し、モーションを検出する"""
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("カメラからのフレーム取得に失敗しました")
                break
            
            # フレームを水平方向に反転して鏡のように表示
            frame = cv2.flip(frame, 1)
            
            # グレースケールに変換
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            if self.prev_frame is None:
                self.prev_frame = gray
                continue
            
            # フレーム間の差分を計算
            frame_delta = cv2.absdiff(self.prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # 画像を膨張させて穴を埋める
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # 輪郭を見つける
            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_amount = np.sum(thresh) / 255  # 動きの量
            self.motion_history.append(motion_amount)
            
            # 動きのある部分を検出して追跡
            self.motion_points = []
            for c in contours:
                if cv2.contourArea(c) < self.motion_threshold:
                    continue
                
                (x, y, w, h) = cv2.boundingRect(c)
                center_x = x + w // 2
                center_y = y + h // 2
                self.motion_points.append((center_x, center_y, w * h))  # 位置と面積
                
                # 動きのある部分を四角で囲む
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # 動きの平均量に基づいてライトの強度を計算
            avg_motion = sum(self.motion_history) / len(self.motion_history) if self.motion_history else 0
            normalized_motion = min(1.0, avg_motion / 50000)  # 正規化（0.0-1.0の範囲に）
            
            # 動きの強さに応じて色を変更
            if normalized_motion > 0.7:  # 激しい動き
                self.light_color = (255, 0, 0)  # 赤
            elif normalized_motion > 0.4:  # 中程度の動き
                self.light_color = (255, 165, 0)  # オレンジ
            elif normalized_motion > 0.1:  # 軽い動き
                self.light_color = (0, 165, 255)  # 青
            else:  # ほとんど動きなし
                self.light_color = (255, 255, 255)  # 白
            
            self.light_intensity = 0.2 + normalized_motion * 0.8  # 明るさも動きに応じて変更
            
            # ライトの色と強度を更新
            self.update_lights()
            
            # 検出結果の表示
            cv2.putText(frame, f"Motion: {normalized_motion:.2f}", (10, 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            cv2.imshow("Motion Detection", frame)
            cv2.imshow("Threshold", thresh)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                self.running = False
                break
            
            self.prev_frame = gray
    
    def update_lights(self):
        """モーションポイントに基づいてライトを更新"""
        # 動きが大きい場合、すべてのライトの強度を上げ、色を変更
        for light in self.lights:
            light["intensity"] = self.light_intensity
            light["color"] = self.light_color
        
        # 特定の動きポイントに近いライトは強度と色を特別に変更
        for x, y, area in self.motion_points:
            # 画面の比率を調整（カメラ640x480 -> ライト画面800x600）
            scaled_x = int(x * self.screen_width / 640)
            scaled_y = int(y * self.screen_height / 480)
            
            for light in self.lights:
                # 動きポイントとライトの距離を計算
                dist = np.sqrt((light["pos"][0] - scaled_x)**2 + (light["pos"][1] - scaled_y)**2)
                if dist < light["radius"] * 2:  # ライトの影響範囲内
                    # 距離が近いほど強度を高く
                    intensity_boost = max(0, 1 - dist / (light["radius"] * 2))
                    light["intensity"] = min(1.0, self.light_intensity + intensity_boost * 0.5)
                    
                    # 領域によって色を少し変化させる（個性を出す）
                    r, g, b = self.light_color
                    variation = hash(str(light["pos"])) % 50 - 25  # -25から+25の変動
                    light["color"] = (
                        max(0, min(255, r + variation)),
                        max(0, min(255, g + variation)),
                        max(0, min(255, b + variation))
                    )
    
    def render_lights(self):
        """ライトの描画処理"""
        clock = pygame.time.Clock()
        
        while self.running:
            self.screen.fill((0, 0, 0))  # 黒背景
            
            # ライトを描画
            for light in self.lights:
                # 強度に応じてサーフェスを描画
                radius = int(light["radius"] * (0.8 + light["intensity"] * 0.5))
                surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
                
                # 光のグラデーションを作成
                color = light["color"]
                for i in range(radius, 0, -1):
                    alpha = int(255 * (i / radius) * light["intensity"])
                    pygame.draw.circle(surf, (*color, alpha), (radius, radius), i)
                
                # 画面に描画
                self.screen.blit(surf, (light["pos"][0] - radius, light["pos"][1] - radius))
            
            # モーションポイントの描画（デバッグ用）
            for x, y, area in self.motion_points:
                # 画面の比率を調整
                scaled_x = int(x * self.screen_width / 640)
                scaled_y = int(y * self.screen_height / 480)
                size = max(5, min(20, int(np.sqrt(area) / 10)))
                pygame.draw.circle(self.screen, (0, 255, 0), (scaled_x, scaled_y), size)
            
            # 情報テキストの表示
            font = pygame.font.SysFont(None, 24)
            intensity_text = font.render(f"Light Intensity: {self.light_intensity:.2f}", True, (255, 255, 255))
            self.screen.blit(intensity_text, (10, 10))
            
            color_name = "White"
            if self.light_color == (255, 0, 0):
                color_name = "Red (High Motion)"
            elif self.light_color == (255, 165, 0):
                color_name = "Orange (Medium Motion)"
            elif self.light_color == (0, 165, 255):
                color_name = "Blue (Low Motion)"
            
            color_text = font.render(f"Light Color: {color_name}", True, (255, 255, 255))
            self.screen.blit(color_text, (10, 40))
            
            instruction_text = font.render("Press 'Q' to quit", True, (255, 255, 255))
            self.screen.blit(instruction_text, (10, self.screen_height - 30))
            
            pygame.display.flip()
            clock.tick(30)  # 30 FPS
    
    def cleanup(self):
        """リソースの解放"""
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    try:
        system = MotionActivatedLighting()
        system.start()
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()