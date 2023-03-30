import win32gui
import win32con
import win32api
import win32ui
import time


class DesktopDrawing:
    def __init__(self, x=0, y=0):
        # 获取桌面窗口句柄
        self.hwnd = win32gui.GetDesktopWindow()

        # 获取屏幕宽高
        self.width = win32api.GetSystemMetrics(win32con.SM_CXVIRTUALSCREEN)
        self.height = win32api.GetSystemMetrics(win32con.SM_CYVIRTUALSCREEN)

        # 创建设备上下文
        self.hdc = win32gui.GetWindowDC(self.hwnd)
        self.mfc = win32ui.CreateDCFromHandle(self.hdc)

        # 创建内存设备上下文
        self.memdc = self.mfc.CreateCompatibleDC()

        # 创建位图对象
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.mfc, self.width, self.height)

        # 将位图对象选入内存设备上下文
        self.memdc.SelectObject(self.bitmap)

        # 设置画笔颜色和宽度
        self.pen = win32ui.CreatePen(win32con.PS_SOLID, 5, win32api.RGB(255, 0, 0))
        self.memdc.SelectObject(self.pen)

        # 在屏幕上绘制一条线
        self.memdc.MoveTo((x, y))
        self.memdc.LineTo((x + 50, y + 50))

        # 将位图对象拷贝到屏幕设备上下文
        self.mfc.BitBlt((0, 0), (self.width, self.height), self.memdc, (0, 0), win32con.SRCCOPY)

        # 释放设备上下文和位图对象
        self.memdc.DeleteDC()
        self.bitmap.DeleteObject()
        self.mfc.DeleteDC()
        win32gui.ReleaseDC(self.hwnd, self.hdc)


if __name__ == '__main__':
    # 创建DesktopDrawing对象
    dd = DesktopDrawing(100, 100)

    # 休眠5秒钟
    time.sleep(5)
