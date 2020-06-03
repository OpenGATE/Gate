/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

/*
  \class  GateImageWithStatistic
  \author thibault.frisson@creatis.insa-lyon.fr
          laurent.guigues@creatis.insa-lyon.fr
	  david.sarrut@creatis.insa-lyon.fr
*/


#ifndef GATEIMAGEWITHSTATISTIC_HH
#define GATEIMAGEWITHSTATISTIC_HH

#include "GateImage.hh"

//-----------------------------------------------------------------------------
/// \brief
class GateImageWithStatistic
{
 public:

  //-----------------------------------------------------------------------------
  // constructor - destructor
  GateImageWithStatistic();
  virtual ~GateImageWithStatistic();

  // void SetLastHitEventImage(GateImage * lastHitEventImage) { mLastHitEventImage = lastHitEventImage; }
  void SetResolutionAndHalfSize(const G4ThreeVector & resolution, const G4ThreeVector & halfSize);
  void SetResolutionAndHalfSize(const G4ThreeVector & resolution, const G4ThreeVector & halfSize, const G4ThreeVector & position);
  void SetResolutionAndHalfSizeCylinder(const G4ThreeVector & resolution, const G4ThreeVector & halfSize);
  void SetResolutionAndHalfSizeCylinder(const G4ThreeVector & resolution, const G4ThreeVector & halfSize, const G4ThreeVector & position);

  void Allocate();
  void Reset(double val=0.0);

  void AddTempValue(const int index, double value);
  void AddValueAndUpdate(const int index, double value);
  void AddValue(const int index, double value);

  double GetValue(const int index);
  void  SetValue(const int index, double value );
  void Fill(double value);

  void EnableSquaredImage(bool b)     { mIsSquaredImageEnabled = b; }
  void EnableUncertaintyImage(bool b) { mIsUncertaintyImageEnabled = b; }
  void SetScaleFactor(double s);
  void SetNormalizeToMax(bool b)      { mNormalizedToMax = b; mNormalizedToIntegral = !b; }
  void SetNormalizeToIntegral(bool b) { mNormalizedToMax = !b; mNormalizedToIntegral = b; }

  void SetFilename(G4String f);
  void SaveData(int numberOfEvents, bool normalise=false);

  inline G4double GetVoxelVolume() const { return mValueImage.GetVoxelVolume(); }

  virtual void UpdateImage();
  virtual void UpdateSquaredImage();
  virtual void UpdateUncertaintyImage(int numberOfEvents);

  GateVImage & GetValueImage() { return mValueImage; }
  GateVImage & GetUncertaintyImage() { return mUncertaintyImage; }

  void SetOrigin(G4ThreeVector v);
  void SetOverWriteFilesFlag(bool b) { mOverWriteFilesFlag = b; }
  void SetTransformMatrix(const G4RotationMatrix & m);

  protected:
  GateImageDouble mValueImage;
  GateImageDouble mSquaredImage;
  GateImageDouble mTempImage;
  GateImageDouble mUncertaintyImage;
  GateImageDouble mScaledValueImage;
  GateImageDouble mScaledSquaredImage;
  bool mOverWriteFilesFlag;
  bool mNormalizedToMax;
  bool mNormalizedToIntegral;

  bool mIsSquaredImageEnabled;
  bool mIsUncertaintyImageEnabled;
  bool mIsValuesMustBeScaled;

  double mScaleFactor;

  G4String mFilename;
  G4String mSquaredFilename;
  G4String mUncertaintyFilename;

  G4String mInitialFilename;
  G4String mSquaredInitialFilename;
  G4String mUncertaintyInitialFilename;

  int mValueFD;
  int mSquaredFD;
  int mUncertaintyFD;

}; // end class GateImageWithStatistic

#endif /* end #define GATEIMAGEWITHSTATISTIC_HH */
