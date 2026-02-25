class chunks:
    @staticmethod
    def get_chunks():
        """
        Trả về danh sách các video chunk mẫu.
        Mỗi chunk gồm:
          - caption   : mô tả nội dung đoạn video (do VLM sinh ra)
          - timestamp : mốc thời gian bắt đầu-kết thúc
          - video_id  : ID video (dùng để trace lại nguồn)
          - camera_id : ID camera quay đoạn đó
        """
        return [
            {
                "caption": "Công nhân bắt đầu vào ca, mặc đồ bảo hộ và kiểm tra danh sách chấm công tại cửa ra vào",
                "timestamp": "00:00-00:30",
                "video_id": "factory_cam01_20240101",
                "camera_id": "CAM_01"
            },
            {
                "caption": "Xe nâng (forklift) vận chuyển các kiện hàng nguyên liệu từ kho vào khu vực sản xuất chính",
                "timestamp": "00:31-01:15",
                "video_id": "factory_cam01_20240101",
                "camera_id": "CAM_01"
            },
            {
                "caption": "Nhóm kỹ thuật viên đang tập trung quanh máy CNC để hiệu chỉnh thông số kỹ thuật",
                "timestamp": "01:16-02:00",
                "video_id": "factory_cam01_20240101",
                "camera_id": "CAM_02"
            },
            {
                "caption": "Dây chuyền lắp ráp hoạt động ổn định, các công nhân thao tác gắn chip lên bảng mạch",
                "timestamp": "02:01-03:00",
                "video_id": "factory_cam01_20240101",
                "camera_id": "CAM_02"
            },
            {
                "caption": "Nhân viên quản lý chất lượng (QC) kiểm tra ngẫu nhiên các sản phẩm trên băng chuyền bằng kính hiển vi",
                "timestamp": "03:01-03:45",
                "video_id": "factory_cam01_20240101",
                "camera_id": "CAM_03"
            },
            {
                "caption": "Một công nhân đang dọn dẹp khu vực đóng gói và dán nhãn lên các thùng carton thành phẩm",
                "timestamp": "03:46-04:30",
                "video_id": "factory_cam01_20240101",
                "camera_id": "CAM_03"
            },
            {
                "caption": "Đội bảo trì thực hiện kiểm tra định kỳ các đường ống khí nén dọc hành lang xưởng",
                "timestamp": "04:31-05:00",
                "video_id": "factory_cam01_20240101",
                "camera_id": "CAM_04"
            }
        ]