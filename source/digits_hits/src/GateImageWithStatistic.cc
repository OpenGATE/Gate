/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/



#ifndef GATEIMAGEWITHSTATISTIC_CC
#define GATEIMAGEWITHSTATISTIC_CC

#include "GateImageWithStatistic.hh"
#include "GateMessageManager.hh"
#include "GateMiscFunctions.hh"

//-----------------------------------------------------------------------------
/// Constructor
GateImageWithStatistic::GateImageWithStatistic()  {
  mIsSquaredImageEnabled = false;
  mIsUncertaintyImageEnabled = false;
  mIsValuesMustBeScaled = false;
  mOverWriteFilesFlag = true;
  mNormalizedToMax = false;
  mNormalizedToIntegral = false;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
/// Destructor
GateImageWithStatistic::~GateImageWithStatistic()  {
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SetOrigin(G4ThreeVector o) {
  mValueImage.SetOrigin(o);
  mSquaredImage.SetOrigin(o);
  mTempImage.SetOrigin(o);
  mUncertaintyImage.SetOrigin(o);
  mScaledValueImage.SetOrigin(o);
  mScaledSquaredImage.SetOrigin(o);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SetTransformMatrix(const G4RotationMatrix & m) {
  mValueImage.SetTransformMatrix(m);
  mSquaredImage.SetTransformMatrix(m);
  mTempImage.SetTransformMatrix(m);
  mUncertaintyImage.SetTransformMatrix(m);
  mScaledValueImage.SetTransformMatrix(m);
  mScaledSquaredImage.SetTransformMatrix(m);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SetScaleFactor(double s) {
  mScaleFactor = s;
  mIsValuesMustBeScaled = true;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SetResolutionAndHalfSize(const G4ThreeVector & resolution,
                                                      const G4ThreeVector & halfSize,
                                                      const G4ThreeVector & position)  {

  mValueImage.SetResolutionAndHalfSize(resolution, halfSize, position);
  if (mIsUncertaintyImageEnabled) {
    mUncertaintyImage.SetResolutionAndHalfSize(resolution, halfSize, position);
    if (!mIsSquaredImageEnabled) {
      mSquaredImage.SetResolutionAndHalfSize(resolution, halfSize, position);
      mTempImage.SetResolutionAndHalfSize(resolution, halfSize, position);
      mScaledSquaredImage.SetResolutionAndHalfSize(resolution, halfSize, position);
    }
  }
  if (mIsSquaredImageEnabled) {
    mSquaredImage.SetResolutionAndHalfSize(resolution, halfSize, position);
    mTempImage.SetResolutionAndHalfSize(resolution, halfSize, position);
    mScaledSquaredImage.SetResolutionAndHalfSize(resolution, halfSize, position);
  }

  mScaledValueImage.SetResolutionAndHalfSize(resolution, halfSize, position);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SetResolutionAndHalfSize(const G4ThreeVector & resolution,
                                                      const G4ThreeVector & halfSize)  {
  G4ThreeVector mPosition = G4ThreeVector(0.0, 0.0, 0.0);
  SetResolutionAndHalfSize(resolution, halfSize, mPosition);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::Allocate() {
  mValueImage.Allocate();
  if (mIsUncertaintyImageEnabled) {
    mUncertaintyImage.Allocate();
    if (!mIsSquaredImageEnabled) {
      mSquaredImage.Allocate();
      mTempImage.Allocate();
      if (mIsValuesMustBeScaled) mScaledSquaredImage.Allocate();
    }
  }
  if (mIsSquaredImageEnabled) {
    mSquaredImage.Allocate();
    mTempImage.Allocate();
    if (mIsValuesMustBeScaled) mScaledSquaredImage.Allocate();
  }
  if (mIsValuesMustBeScaled) mScaledValueImage.Allocate();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::Reset(double val) {
  mValueImage.Fill(val);
  if (mIsUncertaintyImageEnabled) {
    mUncertaintyImage.Fill(0.0);
    if (!mIsSquaredImageEnabled) {
      mSquaredImage.Fill(val*val);
      mTempImage.Fill(0.0);
      if (mIsValuesMustBeScaled) mScaledSquaredImage.Fill(0.0);
    }
  }
  if (mIsSquaredImageEnabled) {
    mSquaredImage.Fill(val*val);
    mTempImage.Fill(0.0);
    if (mIsValuesMustBeScaled) mScaledSquaredImage.Fill(0.0);
  }
  if (mIsValuesMustBeScaled) mScaledValueImage.Fill(0.0);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::Fill(double value) {
  mValueImage.Fill(value);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
double GateImageWithStatistic::GetValue(const int index) {
  return mValueImage.GetValue(index);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SetValue(const int index, double value) {
  mValueImage.SetValue(index, value);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::AddValue(const int index, double value) {
  GateDebugMessage("Actor", 2, "AddValue index=" << index << " value=" << value << Gateendl);
  mValueImage.AddValue(index, value);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::AddTempValue(const int index, double value) {
  GateDebugMessage("Actor", 2, "AddTempValue index=" << index << " value=" << value << Gateendl);
  mTempImage.AddValue(index, value);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::AddValueAndUpdate(const int index, double value) {

  GateDebugMessageInc("Actor", 2, "AddValue and update -- start: "<<mTempImage.GetSize() << Gateendl);
  double tmp = mTempImage.GetValue(index);
  mValueImage.AddValue(index, tmp);
  if (mIsSquaredImageEnabled || mIsUncertaintyImageEnabled) mSquaredImage.AddValue(index, tmp*tmp);
  mTempImage.SetValue(index, value);
  GateDebugMessageDec("Actor", 2, "AddValue and update -- end"<< Gateendl);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SetFilename(G4String f) {
  mFilename = f;
  mSquaredFilename = G4String(removeExtension(f))+"-Squared."+G4String(getExtension(f));
  mUncertaintyFilename = G4String(removeExtension(f))+"-Uncertainty."+G4String(getExtension(f));

  mInitialFilename = mFilename;
  mSquaredInitialFilename = mSquaredFilename;
  mUncertaintyInitialFilename = mUncertaintyFilename;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::SaveData(int numberOfEvents, bool normalise) {

  // Filename
  if (!mOverWriteFilesFlag) {
    mFilename = GetSaveCurrentFilename(mInitialFilename);
    mSquaredFilename = GetSaveCurrentFilename(mSquaredInitialFilename);
    mUncertaintyFilename = GetSaveCurrentFilename(mUncertaintyInitialFilename);
  }

  double factor=1.0;
  if (mIsSquaredImageEnabled || mIsUncertaintyImageEnabled) { UpdateImage(); }
  if (mIsSquaredImageEnabled) { UpdateSquaredImage(); }
  if (mIsUncertaintyImageEnabled) {
    if (!mIsSquaredImageEnabled) UpdateSquaredImage();
    UpdateUncertaintyImage(numberOfEvents);
  }

  if (mIsValuesMustBeScaled == true) {
    factor = mScaleFactor;
  }

  // If normalize, change the scale factor according to max or sum
  if (normalise) {
    mIsValuesMustBeScaled = true;
    double sum = 0.0;
    double max = 0.0;
    GateImageDouble::const_iterator pi = mValueImage.begin();
    GateImageDouble::const_iterator pe = mValueImage.end();
    while (pi != pe) {
      if (*pi > max) max = *pi;
      sum += *pi*factor;
      ++pi;
    }
    if (mNormalizedToMax) SetScaleFactor(factor*1.0/max);
    if (mNormalizedToIntegral) SetScaleFactor(factor*1.0/sum);
  }

  GateMessage("Actor", 1, "Save " << mFilename << " with scaling = "
              << mScaleFactor << "(" << mIsValuesMustBeScaled << ")\n");

  if (!mIsValuesMustBeScaled) {
    mValueImage.Write(mFilename);
    if (mIsSquaredImageEnabled) mSquaredImage.Write(mSquaredFilename);
  }
  else {
    GateImageDouble::iterator po = mScaledValueImage.begin();
    GateImageDouble::iterator pi = mValueImage.begin();
    GateImageDouble::const_iterator pe = mValueImage.end();
    if (mIsSquaredImageEnabled){
      GateImageDouble::iterator pii = mSquaredImage.begin();
      GateImageDouble::iterator poo = mScaledSquaredImage.begin();
      while (pi != pe) {
	*po = (*pi)*mScaleFactor;
	*poo = (*pii)*mScaleFactor*mScaleFactor;
	++pi;
	++po;
	++pii;
	++poo;
      }
      mScaledSquaredImage.Write(mSquaredFilename);
    }
    else {
      while (pi != pe) {
	*po = (*pi)*mScaleFactor;
	++pi;
	++po;
      }
    }
    mScaledValueImage.Write(mFilename);
    SetScaleFactor(factor); // set back previous scaling factor
  }

  if (mIsUncertaintyImageEnabled) mUncertaintyImage.Write(mUncertaintyFilename);
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::UpdateImage() {
  GateImageDouble::iterator pi = mValueImage.begin();
  GateImageDouble::iterator pt = mTempImage.begin();
  GateImageDouble::const_iterator pe = mValueImage.end();
  while (pi != pe) {
    *pi += (*pt);
    ++pt;
    ++pi;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::UpdateSquaredImage() {
  GateImageDouble::iterator pi = mSquaredImage.begin();
  GateImageDouble::iterator pt = mTempImage.begin();
  GateImageDouble::const_iterator pe = mSquaredImage.end();
  while (pi != pe) {
    *pi += (*pt)*(*pt);
    *pt = 0;
    ++pt;
    ++pi;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateImageWithStatistic::UpdateUncertaintyImage(int numberOfEvents)
{
  GateImageDouble::iterator po = mUncertaintyImage.begin();
  GateImageDouble::iterator pi;
  GateImageDouble::iterator pii;
  GateImageDouble::const_iterator pe;

  pi = mValueImage.begin();
  pii = mSquaredImage.begin();
  pe = mValueImage.end();

  int N = numberOfEvents;

  while (pi != pe) {
    double squared = (*pii);
    double mean = (*pi);

    // Ma2002 p1679 : relative statistical uncertainty
    /*	if (mean != 0.0)
     *po = sqrt( (N*squared - mean*mean) / ((N-1)*(mean*mean)) );
     else *po = 1;*/

    // Chetty2006 p1250 : relative statistical uncertainty
    // exactly same than Ma2002
    if (mean != 0.0 && N != 1 && squared != 0.0){
      *po = sqrt( (1.0/(N-1))*(squared/N - pow(mean/N, 2)))/(mean/N);
    }
    else *po = 1;

    /*
    // Ma2002 p1679 : relative statistical uncertainty (estimation)
    if (mean != 0.0)
    *po = sqrt( squared/(mean*mean) );
    else *po = 1;
    */

    /*
    // Walters2002 p2745 : statistical uncertainty
    if (mean != 0.0) {
    *po = sqrt((1.0/((double)N-1.0)) *
    (squared/(double)N - pow(mean/(double)N, 2)));
    }
    else *po = 1.0;
    */
    ++po;
    ++pi;
    ++pii;
  }
}
//-----------------------------------------------------------------------------

#endif /* end #define GATEIMAGEWITHSTATISTIC_CC */
