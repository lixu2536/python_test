# 这是一个示例 Python 脚本。
import os

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
import aspose.threed as a3d


# 3mf 格式转 stl
def mf2stl():
    # 模型路径
    model_path = 'E:\\thingi10k'
    model_list = os.listdir(model_path)  # 检索文件夹下文件名
    # print(model_list)   # ['Barry Bear.stl', 'Cat in Armor.stl', 'machop.stl', 'puppy.stl']

    # 获取目录下的stl文件.
    mf_list = [item for item in model_list if item.endswith('.3MF') or item.endswith('.3mf')]
    # print(stl_list)  # ['Barry Bear.stl', 'Cat in Armor.stl', 'machop.stl', 'puppy.stl']

    # 渲染图片保存路径
    render_path = 'E:\\thingi10k_STL'
    folder = os.path.exists(render_path)
    if not folder:  # 判断文件夹是否存在，并创建。
        os.makedirs(render_path)

    # 根据模型文件依次导入、渲染（保存）、旋转、渲染（保存）、移除物体。
    for item in mf_list:
        mf_path = os.path.join(model_path, item)
        # print(stl_path)  # E:\blender study\model\Barry Bear.stl

        obj_name = os.path.splitext(item)[0]
        # print(obj_name)  # Barry Bear

        # # 使用numpy-stl库查询stl模型的实际尺寸参数
        # in_mesh = mesh.Mesh.from_file(stl_path)
        # # volume, cog, inertia = in_mesh.get_mass_properties()
        # # #  体积，重心，重心的惯性矩阵
        # xyz = (in_mesh.max_ - in_mesh.min_)
        # sizel = round(xyz[0], 2)
        # sizew = round(xyz[1], 2)
        # sizeh = round(xyz[2], 2)
        # # print(sizel)  # 85.6m X

        # 3mf 文件转为 stl格式
        scene = a3d.Scene.from_file(mf_path)

        # 保存路径
        save_name = render_path + '//' + obj_name + '.stl'
        scene.save(save_name)
        print("save {} done".format(obj_name))


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    mf2stl()

# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
