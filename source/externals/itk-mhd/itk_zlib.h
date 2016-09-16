/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile: itk_zlib.h,v $
  Language:  C++
  Date:      $Date: 2006-09-28 13:11:06 $
  Version:   $Revision: 1.1 $

  Copyright (c) Insight Software Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

  Portions of this code are covered under the VTK copyright.
  See VTKCopyright.txt or http://www.kitware.com/VTKCopyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#ifndef itk_zlib_h
#define itk_zlib_h

/* Use the zlib library configured for ITK.  */
#include "itkThirdParty.h"
#ifdef ITK_USE_SYSTEM_ZLIB
# include <zlib.h>
#else
# include <itkzlib/zlib.h>
#endif

#endif
