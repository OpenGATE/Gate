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
 *  @file GateGammaSourceModels.cc
 */

#include "GateGammaSourceModels.hh"
/** Below add #include with your model class header file.
 * Please add new to #include comment with shot class description.
 * */
#include "GateGammaSourceModelParaPositroniumDecay.hh" //Generate 2 gamma from pPs decay
#include "GateGammaSourceModelOrthoPositroniumDecay.hh" //Generate 3 gamma from oPs decay
#include "GateGammaSourceModelSingleGamma.hh" //Generate 1 gamma
#include "GateGammaSourceModelParaPlusPromptDecay.hh"//Generate 2 gammas from pPs decay and one from deexcitation
#include "GateGammaSourceModelOrthoPlusPromptDecay.hh"//Generate 3 gammas from pPs decay and one from deexcitation

GateGammaSourceModels::GateGammaSourceModels() {}

GateGammaSourceModels::~GateGammaSourceModels() {}

void GateGammaSourceModels::InitModels()
{
 //Below add your class constructor call and description

 //Generate 2 gamma from pPs decay
 GateGammaSourceModelParaPositroniumDecay::GetInstance();

 //Generate 3 gamma from oPs decay
 GateGammaSourceModelOrthoPositroniumDecay::GetInstance();

 //Generate single gamma - useful for tests
 GateGammaSourceModelSingleGamma::GetInstance();

 //Generate 2 + 1 gammas
 GateGammaSourceModelParaPlusPromptDecay::GetInstance();

 //Generate 3 + 1 gammas
 GateGammaSourceModelOrthoPlusPromptDecay::GetInstance();
}

