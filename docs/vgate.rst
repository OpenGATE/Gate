.. _vgate-label:

vGate (virtual Gate)
=====================

.. contents:: Table of Contents
   :depth: 15
   :local:

.. figure:: vgate_9.0_scshot.png
   :alt: Figure 1: vgate_9.0_scshot
   :name: vgate_9.0_scshot

Generalities
------------

What is vGate?
~~~~~~~~~~~~~~~

vGate stands for Virtual Gate. It is a complete virtual machine running an `Ubuntu <http://www.ubuntu.com/>`_ 64 bits operating system and made using the free software `Virtual Box <http://www.virtualbox.org/>`_. This virtual machine can be run on any host machine (Linux, Windows, MacOS, ...) provided the Virtual Box program is installed and ready for use. Note: also install the VirtualBox Extension Pack that provides support for USB 2.0 and USB 3.0 devices, and will allow you to create a shared folder between your computer and vGate.

With vGate you can launch your first GATE simulation in just a few steps! No need to install anything, no need to configure anything, and no time spent to understand compilation and related stuff. A full Linux environment is totally set up to be able to use GATE just by launching a simple command: "Gate".

The following software is installed on this machine (see this `file <https://github.com/OpenGATE/Gate/blob/develop/source/docker/vGate.sh>`_ for more details):

* Ubuntu LTS 22.04 on Virtual Box (40GB virtual HD)
* GATE 9.3
* Geant4 11.1.1
* GateContrib: a user-oriented public repository of Gate (macros, examples and user contributions)
* Root 6.26.08
* libtorch cxx11 (cpu) 1.7.0
* ITK v5.3.0 with Module_RTK=ON
* VTK v9.0.3
* vV 1.4
* ImageJ (Fiji)

How to get vGATE now?
~~~~~~~~~~~~~~~~~~~~~

Go to the `OpenGate collaboration website <http://opengatecollaboration.org/>`_. You will then be able to download the virtual machine under the "Download/vGATE" menu. The download link can also be found `here <http://dl.opengatecollaboration.org/vgate>`_.

Be aware that the file you will download a pretty big (about 15 GB), so if several users are downloading the file at the same time, your download speed will be limited and you will have to be patient.

Using a download manager with resume capability (like wget).

How to use vGATE now?
~~~~~~~~~~~~~~~~~~~~~

As the vGate machine has been built using the Virtual Box software, you will have to install this software on your host machine first. And since the version of Virtual Box used to build vGate was the release 6.0.18, you have to install at least this version to be able to run the virtual machine.

Once Virtual Box is installed, here are the steps to get your virtual machine working:

* Launch Virtual Box and in the File menu, select Import Appliance. The Appliance Import wizard is displayed in a new window.
* Click on the small yellow folder icon 'Choose', browse to the location containing the *.ovf or *.ova file of the virtual machine you want to import, and click Open. The Appliance Import Settings step will display.
* Click Import. The Appliance Import Wizard is closed and after a few moments, the imported virtual machine will be listed in Oracle VM VirtualBox Manager.
* After the import, select the imported virtual machine and in the toolbar click the Settings button. Review the virtual machine settings to make sure that the virtual machine has the hardware it needs to operate. You can adjust the number of CPUs and the RAM you want to give to the VM.
* Once you have reviewed the settings, select the imported virtual machine and in the toolbar click the Start button.

That's it!

Login credentials are (qwerty keyboard):

* user: vgate
* password: virtual
* ('gate' is sudo)

How can I find and launch GATE in vGate?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Everything is already configured in the virtual machine to be able to launch GATE without any difficulty. If you want to know how the machine has been configured, you can find all information inside the virtual machine.

Once you start the virtual machine, you can launch the web browser Firefox. Firefox is directly showing the documentation pages (in HTML) that are inside the virtual machine. So please refer to this documentation.

To launch Gate simply open a terminal and type::

   Gate

Any additional questions can be posted on the gate-users mailing-list.

Miscellaneous
-------------

How to share a folder?
~~~~~~~~~~~~~~~~~~~~~~

1. Share a folder on the host OS

• In VirtualBox, click your OS on the left and click on Settings.
• Click on the Shared Folders tab.
• Click on the folder with the plus on the right.
• Browse to a folder of your choice in the folder path.
• Enter a folder name with no spaces e.g. “Share”.
• Check Auto-mount and Make Permanent, if available.
• Click on OK.

2. Mount the folder in the guest OS

• Create a folder in your guest OS that you want to share, like mkdir /media/Share
• Open up Terminal.
• Switch to the root user using sudo su and enter your password.
• Browse to the etc folder using cd /etc.
• Edit the rc.local file using vi fstab
• Move your cursor to the last line and press the letter “i” on your keyboard to insert text.
• Type in the following: Share /media/Share vboxsf  uid=1000,gid=1000 0 0 
• Share should match the folder name from step 1.
• username should match your Linux username.
• /media/Share should be the absolute path of the new folder you created.
• Now hit “ESC”, type :wq and hit ENTER to save and quit the file editing.

After you restart the guest OS, your shared folder will be automatically mounted.




How to get my keyboard properly working?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

As the keyboard type is automatically detected during the Ubuntu installation, it is for the moment adapted for the person how build the virtual machine! It could be annoying.

So if you want your keyboard to work properly, proceed as follows:

* Go into the "System" menu, then in "Preferences" and finally in "Keyboard".
* Go in the "Layout" tab and choose the appropriate layout corresponding to your keyboard.

It should work now.

How to get the network working in my virtual machine?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default settings should just work fine.

There is several ways to get a network connection in the virtual machine. This strictly depends on the characteristics of the network of the host machine (public network, private network, dhcp server policy, dynamic IP, static IP, ...). So ask your network administrator or yourself if your are the administrator.

Once you get this information, then you can read the Virtual Box documentation concerning the `network section <http://www.virtualbox.org/manual/ch03.html#settings-network>`_, or at least see the proposed solutions in the machine settings menu. As they say: "In most cases, this default setup will work fine for you."!

How to transfer files from the virtual machine to my actual machine?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are several solutions:

* Configure a shared directory between the host and the guest machine. This is explained in the `Virtual Box documentation <http://www.virtualbox.org/wiki/Documentation>`_, so please read this `documentation section on Folder Sharing <http://www.virtualbox.org/manual/ch04.html#sharedfolders>`_.
* In case of a connection on a network including machines that you own, you can establish a NFS (`Network File System <http://nfs.sourceforge.net/>`_) to be able to mount an existing filesystem of another machine in your virtual machine. Again you can read documentation on that by searching for NFS (`documentation for Ubuntu <https://help.ubuntu.com/community/SettingUpNFSHowTo>`_).
* If you have an internet connection, you can use FTP access (using `FileZilla <http://filezilla-project.org/>`_ for example) on an external FTP server on which you have access.
* At least you can send your files via email!

How to minimize the size of the VDI (Virtual Disk Image)?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First you have to force a *fsck* (FileSystem Check) of your guest system to have all data arranged at the beginning of the virtual disk. To do that you have to create an empty file named "forcefsck" at the root level (/), using::

  sudo touch /forcefsck

Then you can reboot the virtual machine and the *fsck* will be forced at the boot time. Depending on the space used in your disk, it can take some time.

Once the machine is rebooted, we have to fill all remaining free space with 0 (zero) value. To do this, just run the following command until there is no free space at all::

  sudo dd if=/dev/zero of=/dd_zero_file

It can take a while because it will create a file with the size of the total free space before you run the command.

**Be aware that the size of the VDI of the virtual machine in your host machine will grow too ! (but not necessarily linearly)'**

**It will grow to the maximum allowed size of the dynamic VDI (default is 20Gbytes).**

**So check your free space.**

Once it is done, just remove the created "dd_zero_file" file and shutdown the virtual machine and also the Virtual Box program. Then in your host system, just open a terminal, go in the directory where your VDI file is, and use the following command to finally compress your VDI file::

  sudo VBoxManage modifyvdi /absolute/path/to/your/image.vdi compact

It will also take a while, but after that, your VDI file will be smaller than initially.

How to update the maximum allowed size of my VDI to a bigger one?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To do that, the trick is to do as if you wanted to add a new physical hard drive disk (HDD) to your computer. Every step will be the same except that instead of adding a real HDD, we will add a virtual HDD.

Here are the steps to have more space into your virtual machine:

* The first step is to create a new virtual disk image (VDI). To do that go in Virtual Box in the "File" menu and click on "Virtual Media Manager". Click on "New" to create the VDI, choose a dynamic disk, give it a name, a size, and click on "Finish".
* Then shutdown your machine if it is running, and go into the "Settings" menu. Go into the "Storage" section and click on the "Add Hard Disk" icon. And add your new VDI that you have just created (automatically done in most cases).
* Now turn your virtual machine on. And open a terminal.
* Type the following command::

    ls -l /dev/sd*

  You will see your new device that appears under a name *sdX*, where *X* will be the next letter in alphabetical order after the last disk you inserted in your system. So if it is the first time you do that, your disk will be *sdb*.

* The next step is to create a partition in this new disk. We will use the *fdisk* program. So type the following command::

    sudo fdisk /dev/sdX (where X is the appropriate letter of your disk)

* Then in the *fdisk* menu, you can type **m** to get the list of commands. In our case, type **n** to create a new partition, select 'primary partition' as number 1. Then let the default values to get a full partition on the whole disk.

* Once it is done, type **w** to write the partition table. The program *fdisk* will exit on finish.

* Now you have to format your new partition. This partition appears in *dev/* as *sdX1*. To do that, use the following command::

    sudo mkfs.ext4 /dev/sdX1 (again where X is the appropriate letter of your disk)

* Your disk is ready for use, you just have to mount it somewhere to use it. For example if we want to have this disk in */mnt/* (usual way to do) with the name *my_new_disk*, proceed as follows::

    sudo mkdir /mnt/my_new_disk (to create the directory where the disk will be mounted)
    sudo mount /mnt/sdX1 /mnt/my_new_disk/ (to mount the disk in the directory)

* It is done! You can access and use your new disk in */mnt/my_new_disk*. You can type the command *df* to see your new disk is here.

* Also if you want your new disk to be automatically mounted each time you reboot your machine, you have to add an entry in the file */etc/fstab*. **Be careful as this file is very sensitive to mistakes, your system can be hard to repair if you modify existing lines or introduce mistakes in it!**

* But here is the line to add in this file to have an automated mount of your disk::

    /dev/sdX1   /mnt/my_new_disk  ext4   defaults   0   3

* Of course do not forget to replace the *sdX1* by the appropriate name of your partition, and also for *my_new_disk* is you choose to give it another name. *ext4* is the type of the file system used here.

* On next reboot your disk will be automatically mounted.


