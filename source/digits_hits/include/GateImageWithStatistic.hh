/*----------------------
   GATE version name: gate_v6

   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
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

  void SetFilename(G4String f);
  void SaveData(int numberOfEvents, bool normalise=false);

  inline G4double GetVoxelVolume() const { return mValueImage.GetVoxelVolume(); }

  virtual void UpdateImage();
  virtual void UpdateSquaredImage();
  virtual void UpdateUncertaintyImage(int numberOfEvents);

  GateImage & GetValueImage() { return mValueImage; }

  protected:
  GateImage mValueImage;
  GateImage mSquaredImage;
  GateImage mTempImage;
  GateImage mUncertaintyImage;
  GateImage mScaledValueImage;
  GateImage mScaledSquaredImage;   
 // GateImage * mLastHitEventImage;

  bool mIsSquaredImageEnabled;
  bool mIsUncertaintyImageEnabled;
  bool mIsValuesMustBeScaled;
  
  double mScaleFactor;

  G4String mFilename;
  G4String mSquaredFilename;
  G4String mUncertaintyFilename;

  int mValueFD;
  int mSquaredFD;
  int mUncertaintyFD;



}; // end class GateImageWithStatistic

#endif /* end #define GATEIMAGEWITHSTATISTIC_HH */

