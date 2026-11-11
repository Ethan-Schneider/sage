from setuptools import find_packages, setup

package_name = 'sage_base_teleop'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ethan',
    maintainer_email='eschneider32@gatech.edu',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
            'console_scripts': [
                    'teleop_publisher = sage_base_teleop.teleop_control:main',
                    'teleop_subscriber = sage_base_teleop.teleop_actuate:main'
            ],
    },
)
