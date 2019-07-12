/**
 *  @copyright Copyright 2016 The J-PET Gate Authors. All rights reserved.
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 *  GNU General Public License for more details.
 *
 *  @fileGateGammaSourceModels.hh
 */

#ifndef GateGammaSourceModels_hh
#define GateGammaSourceModels_hh


/**Author: Mateusz Bała
 * Email: bala.mateusz@gmail.com
 * About class: This class fill GateJPETSourceManager with informations about models classes. When you create new model class which have to be included in GateJPETSourceManager add new line in function InitModels().
 * */
class GateGammaSourceModels
{
 public:
  GateGammaSourceModels();
  ~GateGammaSourceModels();

  /** Generate all models once. Place in this function body your class GetInstance() method call.* */
  static void InitModels();
};

#endif
