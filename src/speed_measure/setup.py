from setuptools import setup

package_name = 'speed_measure'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'image_undistort_node = speed_measure.nodes.image_undistort_node:main',
            'optical_flow_node = speed_measure.nodes.optical_flow_node:main',
            'filter_node = speed_measure.nodes.median_filter_node:main'
        ],
    },
)
