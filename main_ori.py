import sys
import os
import tempfile
import traceback
import fitz  # PyMuPDF
import uuid
from PyQt5 import QtCore, QtGui, QtWidgets
from paddleocr import PaddleOCR
from ocrUI import Ui_MainWindow

# ==========================================
# 全域宣告 OCR 引擎
# ==========================================
GLOBAL_OCR_ENGINE = None

# 🚀 負責在程式啟動時，於背景載入模型的執行緒
class ModelInitWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal()
    error = QtCore.pyqtSignal(str)

    def run(self):
        try:
            global GLOBAL_OCR_ENGINE
            if GLOBAL_OCR_ENGINE is None:
                print("⏳ 程式啟動，正在背景初始化 PaddleOCR 模型 (GPU 模式)...")
                
                try:
                    # 嘗試加入 CSDN 建議的最佳化參數 (移除已知會報錯的 det_db_score_mode)
                    print("🔄 嘗試使用進階參數載入模型...")
                    GLOBAL_OCR_ENGINE = PaddleOCR(
                        use_textline_orientation=True, 
                        lang="ch", 
                        device="gpu",
                        det_db_unclip_ratio=2.0,      # 適當擴大檢測框，防止邊緣被切掉
                        rec_batch_num=1               # 關閉批次處理避免錯位
                    )
                except ValueError as ve:
                    # 🚀 安全回退機制：如果版本不支援上述參數，自動退回最基本的安全設定
                    print(f"⚠️ 進階參數不相容 ({ve})，自動回退至基本安全設定...")
                    GLOBAL_OCR_ENGINE = PaddleOCR(
                        use_textline_orientation=True, 
                        lang="ch", 
                        device="gpu"
                    )
                
                print("✅ PaddleOCR 模型初始化成功！")
            self.finished.emit()
        except Exception as e:
            print("\n" + "="*60)
            print("❌ 模型初始化發生崩潰：")
            traceback.print_exc()
            print("="*60 + "\n")
            self.error.emit(str(e))

class OCRWorker(QtCore.QThread):
    finished = QtCore.pyqtSignal(list)
    error = QtCore.pyqtSignal(str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        try:
            global GLOBAL_OCR_ENGINE
            if GLOBAL_OCR_ENGINE is None:
                raise RuntimeError("模型尚未初始化完成！")
            
            print(f"🔍 開始辨識圖片: {self.image_path}")
            results = GLOBAL_OCR_ENGINE.predict(self.image_path)
            print("✅ 辨識完成！正在解析結果...")
            
            parsed_results = []
            result_list = list(results)
            
            if result_list and len(result_list) > 0:
                res_obj = result_list[0]
                
                res_dict = {}
                if isinstance(res_obj, dict): res_dict = res_obj
                elif hasattr(res_obj, 'res') and isinstance(res_obj.res, dict): res_dict = res_obj.res
                elif hasattr(res_obj, '__dict__'): res_dict = res_obj.__dict__
                
                boxes = res_dict.get('dt_polys') or res_dict.get('polys') or res_dict.get('boxes') or []
                texts = res_dict.get('rec_text') or res_dict.get('texts') or res_dict.get('rec_texts') or []
                
                if boxes and texts and len(boxes) == len(texts):
                    for i in range(len(boxes)):
                        box = boxes[i]
                        text = texts[i]
                        parsed_results.append({
                            "poly": box,  # 直接保留原始的 4 個頂點座標
                            "text": text
                        })
                elif not boxes and not texts and isinstance(res_obj, list) and len(res_obj) > 0:
                    try:
                        for line in res_obj:
                            if len(line) == 2 and isinstance(line[0], list):
                                parsed_results.append({
                                    "poly": line[0], # 保留原始多邊形
                                    "text": line[1][0] if isinstance(line[1], (list, tuple)) else line[1]
                                })
                    except Exception as e:
                        print(f"備用解析方案失敗: {e}")
                else:
                    print("⚠️ 無法解析結果結構或未能辨識出文字。")
            
            self.finished.emit(parsed_results)
            
        except Exception as e:
            print("\n" + "="*60)
            print("❌ OCR 背景執行緒發生崩潰，詳細錯誤訊息如下：")
            traceback.print_exc()
            print("="*60 + "\n")
            self.error.emit(str(e))

# ==========================================
# 以下為 UI 介面與自定義元件
# ==========================================
class DropAreaLabel(QtWidgets.QLabel):
    fileDropped = QtCore.pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(QtCore.Qt.AlignCenter)
        self.setText("\n\n拖移文件至此處\n(支援 PNG, JPG, PDF)")
        self.setStyleSheet("""
            QLabel {
                border: 2px dashed #90CAF9; border-radius: 8px;
                background-color: #E3F2FD; color: #1565C0;
                font-weight: bold; font-size: 14px;
            }
            QLabel:hover { background-color: #BBDEFB; border-color: #1565C0; }
        """)
        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls(): event.accept()
        else: event.ignore()

    def dropEvent(self, event):
        urls = event.mimeData().urls()
        if urls: self.fileDropped.emit(urls[0].toLocalFile())

class BBoxItem(QtWidgets.QGraphicsPolygonItem):
    def __init__(self, polygon, text_index, main_window):
        super().__init__(polygon)
        self.text_index = text_index
        self.main_window = main_window
        
        self.default_pen = QtGui.QPen(QtGui.QColor("#1565C0"), 2)
        self.default_pen.setCosmetic(True)  # 🚀 確保 View 縮放時，外框粗細固定不走樣
        self.default_brush = QtGui.QBrush(QtGui.QColor(21, 101, 192, 50))
        
        self.active_pen = QtGui.QPen(QtGui.QColor("#FF3D00"), 3)
        self.active_pen.setCosmetic(True)   # 🚀 確保 View 縮放時，外框粗細固定不走樣
        self.active_brush = QtGui.QBrush(QtGui.QColor(255, 61, 0, 80))

        self.setPen(self.default_pen)
        self.setBrush(self.default_brush)
        self.setAcceptHoverEvents(True)

    def hoverEnterEvent(self, event): self.setCursor(QtCore.Qt.PointingHandCursor)
    def hoverLeaveEvent(self, event): self.setCursor(QtCore.Qt.ArrowCursor)
    def mousePressEvent(self, event): self.main_window.sync_selection(self.text_index, source='canvas')

    def set_active(self, active):
        if active:
            self.setPen(self.active_pen)
            self.setBrush(self.active_brush)
        else:
            self.setPen(self.default_pen)
            self.setBrush(self.default_brush)

# ==========================================
# 主視窗邏輯
# ==========================================
class OcrMainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.setWindowTitle("AAEON OCR SYSTEM")

        self.ui.frame.setStyleSheet("border: none; background: transparent;")
        self.ui.input.setStyleSheet(self.ui.input.styleSheet() + "border: none;")
        self.ui.overview.setStyleSheet(self.ui.overview.styleSheet() + "border: none;")
        self.ui.text_output.setStyleSheet(self.ui.text_output.styleSheet() + "border: none;")

        self.bbox_items = [] 
        self.current_pdf_doc = None
        self.current_page_index = 0
        self.total_pages = 0
        self.is_pdf = False
        self.temp_img_path = "" 
        
        self._setup_input_area()
        self._setup_overview_area()
        self._setup_text_output_area()

        self._initialize_model_in_background()

    def _initialize_model_in_background(self):
        self.btn_start_ocr.setEnabled(False)
        self.btn_start_ocr.setText("⏳ 模型載入中...請稍候")
        
        self.init_thread = ModelInitWorker()
        self.init_thread.finished.connect(self.on_model_loaded)
        self.init_thread.error.connect(self.on_model_load_error)
        self.init_thread.start()

    def on_model_loaded(self):
        self.btn_start_ocr.setEnabled(True)
        self.btn_start_ocr.setText("開始 OCR 辨識")

    def on_model_load_error(self, err_msg):
        self.btn_start_ocr.setText("❌ 模型載入失敗")
        QtWidgets.QMessageBox.critical(self, "錯誤", f"模型初始化失敗，請查看 Terminal 日誌。\n{err_msg}")

    def _setup_input_area(self):
        layout = self.ui.verticalLayout
        layout.setContentsMargins(15, 20, 15, 20)
        
        self.drop_area = DropAreaLabel()
        self.drop_area.fileDropped.connect(self.load_file)
        self.drop_area.setMinimumHeight(200)
        layout.addWidget(self.drop_area)

        path_layout = QtWidgets.QHBoxLayout()
        self.path_input = QtWidgets.QLineEdit()
        self.path_input.setPlaceholderText("或輸入檔案路徑...")
        self.path_input.setStyleSheet("padding: 5px; border: 1px solid #ccc; border-radius: 4px;")
        
        self.btn_browse = QtWidgets.QPushButton("瀏覽")
        self.btn_browse.setStyleSheet("background-color: #EEEEEE; padding: 5px 15px; border-radius: 4px;")
        self.btn_browse.clicked.connect(self.browse_file)
        
        path_layout.addWidget(self.path_input)
        path_layout.addWidget(self.btn_browse)
        layout.addLayout(path_layout)
        layout.addStretch()

        self.btn_start_ocr = QtWidgets.QPushButton("開始 OCR 辨識")
        self.btn_start_ocr.setMinimumHeight(50)
        self.btn_start_ocr.setStyleSheet("""
            QPushButton {
                background-color: #1565C0; color: white;
                font-size: 16px; font-weight: bold; border-radius: 8px;
            }
            QPushButton:hover { background-color: #0D47A1; }
            QPushButton:pressed { background-color: #002171; }
            QPushButton:disabled { background-color: #9E9E9E; color: #E0E0E0; }
        """)
        self.btn_start_ocr.clicked.connect(self.run_ocr)
        layout.addWidget(self.btn_start_ocr)

    def _setup_overview_area(self):
        layout = QtWidgets.QVBoxLayout(self.ui.overview)
        layout.setContentsMargins(15, 15, 15, 15) 
        
        toolbar_layout = QtWidgets.QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 0, 5, 10) 
        
        title = QtWidgets.QLabel("<b>文件預覽與識別結果</b>")
        title.setStyleSheet("color: #333; font-size: 15px; padding: 5px;")
        toolbar_layout.addWidget(title)
        toolbar_layout.addStretch()

        self.btn_prev_page = QtWidgets.QPushButton("◀ 上一頁")
        self.btn_next_page = QtWidgets.QPushButton("下一頁 ▶")
        self.lbl_page_info = QtWidgets.QLabel("1 / 1")
        
        page_btn_style = """
            QPushButton {
                padding: 6px 12px; background-color: #E3F2FD; color: #1565C0;
                border-radius: 4px; font-weight: bold;
            }
            QPushButton:hover { background-color: #BBDEFB; }
            QPushButton:disabled { background-color: #F5F5F5; color: #BDBDBD; }
        """
        self.btn_prev_page.setStyleSheet(page_btn_style)
        self.btn_next_page.setStyleSheet(page_btn_style)
        self.lbl_page_info.setStyleSheet("padding: 0 15px; font-weight: bold; font-size: 14px; color: #333;")
        
        self.btn_prev_page.setVisible(False)
        self.btn_next_page.setVisible(False)
        self.lbl_page_info.setVisible(False)
        
        self.btn_prev_page.clicked.connect(self.prev_page)
        self.btn_next_page.clicked.connect(self.next_page)
        
        toolbar_layout.addWidget(self.btn_prev_page)
        toolbar_layout.addWidget(self.lbl_page_info)
        toolbar_layout.addWidget(self.btn_next_page)
        layout.addLayout(toolbar_layout)

        self.scene = QtWidgets.QGraphicsScene()
        self.graphics_view = QtWidgets.QGraphicsView(self.scene)
        self.graphics_view.setStyleSheet("border: 1px solid #ddd; background-color: #fafafa; border-radius: 4px;")
        self.graphics_view.setDragMode(QtWidgets.QGraphicsView.ScrollHandDrag)
        layout.addWidget(self.graphics_view)
        
        self.current_pixmap_item = None

    def _setup_text_output_area(self):
        layout = QtWidgets.QVBoxLayout(self.ui.text_output)
        layout.setContentsMargins(15, 15, 15, 15)
        
        toolbar_layout = QtWidgets.QHBoxLayout()
        toolbar_layout.setContentsMargins(5, 0, 5, 10)
        
        self.btn_copy = QtWidgets.QPushButton("📋 複製全文")
        self.btn_copy.setStyleSheet("""
            QPushButton {
                background-color: #E3F2FD; color: #1565C0; 
                border-radius: 4px; padding: 6px 12px; font-weight: bold;
            }
            QPushButton:hover { background-color: #BBDEFB; }
        """)
        self.btn_copy.clicked.connect(self.copy_all_text)
        toolbar_layout.addWidget(self.btn_copy)
        toolbar_layout.addStretch()
        layout.addLayout(toolbar_layout)

        self.text_list = QtWidgets.QListWidget()
        self.text_list.setStyleSheet("""
            QListWidget { border: 1px solid #ddd; font-size: 14px; padding: 5px; border-radius: 4px; }
            QListWidget::item { padding: 8px; border-bottom: 1px solid #eee; }
            QListWidget::item:selected { background-color: #E3F2FD; color: #1565C0; font-weight: bold; border-radius: 4px;}
        """)
        self.text_list.itemClicked.connect(self.on_list_item_clicked)
        layout.addWidget(self.text_list)

    # ==========================================
    # 檔案載入與顯示邏輯
    # ==========================================
    def browse_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "選擇文件", "", "Images/PDF (*.png *.jpg *.jpeg *.pdf)")
        if file_path: self.load_file(file_path)

    def load_file(self, file_path):
        self.path_input.setText(file_path)
        self.scene.clear()
        self.bbox_items.clear()
        self.text_list.clear()

        if file_path.lower().endswith('.pdf'):
            self.is_pdf = True
            if self.current_pdf_doc: self.current_pdf_doc.close()
            self.current_pdf_doc = fitz.open(file_path)
            self.total_pages = len(self.current_pdf_doc)
            self.current_page_index = 0
            
            self.btn_prev_page.setVisible(True)
            self.btn_next_page.setVisible(True)
            self.lbl_page_info.setVisible(True)
            self.render_pdf_page()
        else:
            self.is_pdf = False
            self.btn_prev_page.setVisible(False)
            self.btn_next_page.setVisible(False)
            self.lbl_page_info.setVisible(False)
            
            pixmap = QtGui.QPixmap(file_path)
            self.display_pixmap(pixmap)

    def render_pdf_page(self):
        if not self.current_pdf_doc: return
        self.lbl_page_info.setText(f"{self.current_page_index + 1} / {self.total_pages}")
        self.btn_prev_page.setEnabled(self.current_page_index > 0)
        self.btn_next_page.setEnabled(self.current_page_index < self.total_pages - 1)

        page = self.current_pdf_doc.load_page(self.current_page_index)
        mat = fitz.Matrix(2.0, 2.0)
        pix = page.get_pixmap(matrix=mat)
        
        fmt = QtGui.QImage.Format_RGBA8888 if pix.alpha else QtGui.QImage.Format_RGB888
        qimg = QtGui.QImage(pix.samples, pix.width, pix.height, pix.stride, fmt)
        qpixmap = QtGui.QPixmap.fromImage(qimg)
        
        self.bbox_items.clear()
        self.text_list.clear()
        self.display_pixmap(qpixmap)

    def display_pixmap(self, pixmap):
        if not pixmap.isNull():
            self.scene.clear()
            self.current_pixmap_item = self.scene.addPixmap(pixmap)
            self.scene.setSceneRect(QtCore.QRectF(pixmap.rect()))
            self.graphics_view.fitInView(self.scene.sceneRect(), QtCore.Qt.KeepAspectRatio)

    def prev_page(self):
        if self.current_page_index > 0:
            self.current_page_index -= 1
            self.render_pdf_page()

    def next_page(self):
        if self.current_page_index < self.total_pages - 1:
            self.current_page_index += 1
            self.render_pdf_page()

    # ==========================================
    # PaddleOCR 執行邏輯
    # ==========================================
    def run_ocr(self):
        if not self.current_pixmap_item:
            QtWidgets.QMessageBox.warning(self, "警告", "請先載入文件或圖片！")
            return

        self.btn_start_ocr.setEnabled(False)
        self.btn_start_ocr.setText("辨識中...請稍候")
        self.btn_prev_page.setEnabled(False)
        self.btn_next_page.setEnabled(False)

        temp_dir = tempfile.gettempdir()
        self.temp_img_path = os.path.join(temp_dir, f"ocr_target_{uuid.uuid4().hex}.png")
        
        ui_pixmap = self.current_pixmap_item.pixmap()
        ui_pixmap.toImage().save(self.temp_img_path, "PNG")

        self.ocr_thread = OCRWorker(self.temp_img_path)
        self.ocr_thread.finished.connect(self.on_ocr_finished)
        self.ocr_thread.error.connect(self.on_ocr_error)
        self.ocr_thread.start()

    def on_ocr_finished(self, results):
        self.btn_start_ocr.setEnabled(True)
        self.btn_start_ocr.setText("開始 OCR 辨識")
        
        if self.is_pdf:
            self.btn_prev_page.setEnabled(self.current_page_index > 0)
            self.btn_next_page.setEnabled(self.current_page_index < self.total_pages - 1)

        # 移除舊框，不使用 self.scene.clear() 避免破壞畫布原點
        for bbox in self.bbox_items:
            self.scene.removeItem(bbox)
        self.bbox_items.clear()
        self.text_list.clear()

        if not results:
            QtWidgets.QMessageBox.information(self, "提示", "此頁面未辨識出任何文字。")
            return

        # 🚀 分別計算 X 與 Y 軸的 DPI 縮放比例，防止長寬細微誤差導致座標偏移
        ui_pixmap = self.current_pixmap_item.pixmap()
        img_width = ui_pixmap.toImage().width()
        img_height = ui_pixmap.toImage().height()
        pix_width = ui_pixmap.width()
        pix_height = ui_pixmap.height()
        
        scale_x = img_width / pix_width if pix_width > 0 else 1.0
        scale_y = img_height / pix_height if pix_height > 0 else 1.0

        for i, res in enumerate(results):
            poly_points = res["poly"]
            
            # 建立多邊形物件並進行精準坐標縮放
            qpolygon = QtGui.QPolygonF()
            for pt in poly_points:
                # 🚀 強制轉為 float，並分別套用 scale_x 與 scale_y
                mapped_x = float(pt[0]) / scale_x
                mapped_y = float(pt[1]) / scale_y
                qpolygon.append(QtCore.QPointF(mapped_x, mapped_y))
            
            bbox = BBoxItem(qpolygon, text_index=i, main_window=self)
            bbox.setParentItem(self.current_pixmap_item)
            self.bbox_items.append(bbox)
            
            item = QtWidgets.QListWidgetItem(res["text"])
            self.text_list.addItem(item)

        if os.path.exists(self.temp_img_path):
            try:
                os.remove(self.temp_img_path)
            except Exception as e:
                print(f"無法刪除暫存檔: {e}")

    def on_ocr_error(self, err_msg):
        self.btn_start_ocr.setEnabled(True)
        self.btn_start_ocr.setText("開始 OCR 辨識")
        QtWidgets.QMessageBox.critical(self, "OCR 錯誤", f"模型辨識過程發生錯誤，請查看 Terminal 日誌。")

    # ==========================================
    # 互動工具
    # ==========================================
    def copy_all_text(self):
        if self.text_list.count() == 0: return
        text_items = [self.text_list.item(i).text() for i in range(self.text_list.count())]
        QtWidgets.QApplication.clipboard().setText("\n".join(text_items))
        QtWidgets.QMessageBox.information(self, "成功", "已將全部辨識文字複製到剪貼簿！")

    def sync_selection(self, index, source):
        for bbox in self.bbox_items: bbox.set_active(False)
        if 0 <= index < len(self.bbox_items):
            self.bbox_items[index].set_active(True)
            if source == 'canvas': self.text_list.setCurrentRow(index)
                
    def on_list_item_clicked(self, item):
        self.sync_selection(self.text_list.row(item), source='list')

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = OcrMainWindow()
    window.show()
    sys.exit(app.exec_())