"""
可视化模块

该模块提供实时3D可视化和视频导出功能。

Classes:
    RealtimeVisualizer: 实时3D可视化器
    VideoExporter: 视频导出器
"""

import numpy as np
import matplotlib
# 使用Agg后端避免显示问题
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, Optional
import cv2


class RealtimeVisualizer:
    """
    实时3D可视化器
    
    使用Matplotlib进行实时3D渲染。适合中小规模场景（<10000物体）。
    
    Attributes:
        fig: Matplotlib图形对象
        ax: 3D坐标轴对象
        scatter: 散点图对象
    """
    
    def __init__(
        self,
        world_bounds: Tuple[Tuple[float, float, float], Tuple[float, float, float]],
        figsize: Tuple[int, int] = (10, 10),
        title: str = "GPU Collision Detection Simulation"
    ):
        """
        初始化可视化器
        
        Args:
            world_bounds: 世界边界 ((xmin, ymin, zmin), (xmax, ymax, zmax))
            figsize: 图形大小 (width, height) 英寸
            title: 窗口标题
        """
        # 不使用交互模式（适合headless环境）
        # plt.ion()  # 开启交互模式
        
        self.fig = plt.figure(figsize=figsize)
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # 设置坐标轴范围
        self.ax.set_xlim(world_bounds[0][0], world_bounds[1][0])
        self.ax.set_ylim(world_bounds[0][1], world_bounds[1][1])
        self.ax.set_zlim(world_bounds[0][2], world_bounds[1][2])
        
        # 设置标签 - 移除刻度使其更像场景而非图表
        self.ax.set_xlabel('X', fontsize=8, labelpad=-5)
        self.ax.set_ylabel('Y', fontsize=8, labelpad=-5)
        self.ax.set_zlabel('Z', fontsize=8, labelpad=-5)
        self.ax.set_title(title, fontsize=14, pad=10)
        
        # 简化刻度显示
        self.ax.set_xticks([world_bounds[0][0], 0, world_bounds[1][0]])
        self.ax.set_yticks([world_bounds[0][1], (world_bounds[0][1] + world_bounds[1][1])/2, world_bounds[1][1]])
        self.ax.set_zticks([world_bounds[0][2], 0, world_bounds[1][2]])
        
        # 设置背景颜色和网格
        self.ax.xaxis.pane.fill = True
        self.ax.yaxis.pane.fill = True
        self.ax.zaxis.pane.fill = True
        self.ax.xaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        self.ax.yaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        self.ax.zaxis.pane.set_facecolor((0.95, 0.95, 0.95, 1.0))
        
        # 设置网格样式
        self.ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.3)
        
        # 设置视角 - 从侧面观察，Y轴（高度）朝上
        # elev=15: 从水平面上方15度看（接近水平视角）
        # azim=45: 从侧前方观察（45度角）
        # roll=0: 无滚转
        # 这样可以清晰看到球向下（-Y方向）落到地面（Y=0）
        self.ax.view_init(elev=15, azim=20, roll=90)
        
        # 确保Y轴是竖直方向（高度）
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y (Height)')
        self.ax.set_zlabel('Z')
        
        self.scatter = None
        self.world_bounds = world_bounds
    
    def update(
        self,
        positions: np.ndarray,
        colors: Optional[np.ndarray] = None,
        radii: Optional[np.ndarray] = None,
        info_text: Optional[str] = None
    ) -> None:
        """
        更新显示
        
        Args:
            positions: 物体位置 [N, 3]
            colors: 物体颜色 [N, 3]，可选
            radii: 物体半径 [N]，用于设置点大小，可选
            info_text: 额外信息文本，显示在图上，可选
        """
        # 移除旧的散点图
        if self.scatter is not None:
            self.scatter.remove()
        
        # 计算点大小（基于半径）
        if radii is not None:
            # 将半径映射到合适的点大小（增大以便更清晰可见）
            sizes = (radii * 50) ** 2  # 增大系数使小球更明显
        else:
            sizes = 100
        
        # 绘制新的散点图 - 更真实的渲染效果
        # 保持坐标一致：物理坐标直接映射到matplotlib坐标
        # 物理: X(水平), Y(垂直/高度), Z(深度)
        # Matplotlib: X, Y, Z - 通过view_init调整视角使Y朝上
        self.scatter = self.ax.scatter(
            positions[:, 0],  # X轴
            positions[:, 1],  # Y轴（高度）
            positions[:, 2],  # Z轴
            c=colors if colors is not None else 'blue',
            s=sizes,
            alpha=1.0,  # 完全不透明
            edgecolors='none',  # 移除边缘线使其更圆润
            linewidths=0,
            depthshade=True  # 启用深度阴影使其更立体
        )
        
        # 显示信息文本
        if info_text:
            # 清除旧文本
            for txt in self.ax.texts:
                txt.remove()
            # 添加新文本
            self.ax.text2D(
                0.05, 0.95, info_text,
                transform=self.ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )
        
        # 刷新显示（非交互模式下不需要）
        # plt.draw()
        # plt.pause(0.001)
    
    def close(self) -> None:
        """关闭可视化窗口"""
        plt.close(self.fig)
    
    def save_frame(self, filename: str) -> None:
        """
        保存当前帧为图片
        
        Args:
            filename: 输出文件名
        """
        self.fig.savefig(filename, dpi=150, bbox_inches='tight')


class VideoExporter:
    """
    视频导出器
    
    将仿真结果导出为视频文件（MP4格式）。
    
    Attributes:
        writer: OpenCV视频写入器
        resolution: 视频分辨率
        fps: 视频帧率
    """
    
    def __init__(
        self,
        filename: str,
        fps: int = 60,
        resolution: Tuple[int, int] = (1920, 1080),
        codec: str = 'MJPG'
    ):
        """
        初始化视频导出器
        
        Args:
            filename: 输出视频文件名（应以.avi或.mp4结尾）
            fps: 视频帧率
            resolution: 视频分辨率 (width, height)
            codec: 视频编解码器（'MJPG' for Motion JPEG, 'mp4v' for MPEG-4）
        
        Note:
            使用MJPG编解码器生成.avi文件，然后可以用ffmpeg转换为H.264 MP4
        """
        # 使用MJPEG编解码器（最兼容和可靠）
        # 先生成临时.avi文件，然后转换为MP4
        self.filename = filename
        self.temp_filename = filename.replace('.mp4', '_temp.avi')
        self.use_temp = filename.endswith('.mp4')
        
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        output_file = self.temp_filename if self.use_temp else filename
        
        self.writer = cv2.VideoWriter(output_file, fourcc, fps, resolution)
        
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {output_file}")
        
        self.resolution = resolution
        self.fps = fps
        self.frame_count = 0
        
        print(f"Video exporter initialized: {filename} "
              f"({resolution[0]}x{resolution[1]} @ {fps}fps, codec: MJPG)")
    
    def add_frame_from_matplotlib(self, fig: plt.Figure) -> None:
        """
        从Matplotlib图形添加一帧
        
        Args:
            fig: Matplotlib图形对象
        """
        # 渲染图形到numpy数组
        fig.canvas.draw()
        
        # 兼容新旧Matplotlib版本
        try:
            # 新版本API (Matplotlib 3.8+)
            img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            # 转换RGBA到RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        except AttributeError:
            # 旧版本API (Matplotlib < 3.8)
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        # 调整大小以匹配目标分辨率
        if img.shape[:2][::-1] != self.resolution:
            img = cv2.resize(img, self.resolution)
        
        # 转换颜色空间（Matplotlib使用RGB，OpenCV使用BGR）
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        # 写入帧
        self.writer.write(img)
        self.frame_count += 1
    
    def add_frame(self, img: np.ndarray) -> None:
        """
        直接添加一帧图像
        
        Args:
            img: 图像数组 (height, width, 3)，BGR格式
        """
        # 调整大小以匹配目标分辨率
        if img.shape[:2][::-1] != self.resolution:
            img = cv2.resize(img, self.resolution)
        
        self.writer.write(img)
        self.frame_count += 1
    
    def release(self) -> None:
        """完成导出并释放资源"""
        self.writer.release()
        print(f"Video export completed: {self.frame_count} frames written")
        
        # 如果使用临时文件，转换为H.264 MP4
        if self.use_temp and self.frame_count > 0:
            import subprocess
            import os
            
            try:
                print(f"Converting to H.264 MP4...")
                # 使用ffmpeg转换为H.264编码的MP4
                cmd = [
                    'ffmpeg', '-y',  # 覆盖输出文件
                    '-i', self.temp_filename,  # 输入文件
                    '-c:v', 'libx264',  # 使用H.264编码器
                    '-preset', 'medium',  # 编码预设
                    '-crf', '23',  # 质量（18-28，越低质量越高）
                    '-pix_fmt', 'yuv420p',  # 像素格式（兼容性好）
                    '-loglevel', 'error',  # 只显示错误
                    self.filename
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                
                # 删除临时文件
                os.remove(self.temp_filename)
                print(f"Video converted successfully: {self.filename}")
                
            except subprocess.CalledProcessError as e:
                print(f"Warning: Failed to convert video to H.264: {e}")
                print(f"Temporary AVI file saved as: {self.temp_filename}")
            except Exception as e:
                print(f"Warning: Error during video conversion: {e}")
                print(f"Temporary AVI file saved as: {self.temp_filename}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.release()
    
    def __del__(self):
        """析构函数"""
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.release()
