/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General  Public Licence (LGPL)
See GATE/LICENSE.txt for further details
----------------------*/

/*
  \class  GateImageWithStatisticTLE
  \author brent.huisman@creatis.insa-lyon.fr
*/


#ifndef GATEIMAGEWITHSTATISTICTLE_HH
#define GATEIMAGEWITHSTATISTICTLE_HH

#include "GateImage.hh"

//-----------------------------------------------------------------------------
/// \brief
class GateImageWithStatisticTLE : GateImageWithStatistic
{
 public:
  virtual void UpdateUncertaintyImage(int numberOfEvents);

}; // end class GateImageWithStatisticTLE

//-----------------------------------------------------------------------------
void GateImageWithStatisticTLE::UpdateUncertaintyImage(int numberOfEvents)
{
  GateImageDouble::iterator po = mUncertaintyImage.begin();
  GateImageDouble::iterator pi;
  GateImageDouble::iterator pii;
  GateImageDouble::const_iterator pe;

  //if(mIsValuesMustBeScaled)  pi = mScaledValueImage.begin();
  pi = mValueImage.begin();

  //if(mIsValuesMustBeScaled) pii = mScaledSquaredImage.begin();
  pii = mSquaredImage.begin();

  //if(mIsValuesMustBeScaled)  pe = mScaledValueImage.end();
  pe = mValueImage.end();

  int N = numberOfEvents;   //NOTE: Not sure if I should use this?

  while (pi != pe) {
    double squared = (*pii);
    double mean = (*pi);  //is not actually mean, has still to be divided by N. its just the value.

    // TLE Magic, see JM paper 2014.
    // His script uses this:
    // varianceL = (SquaredTrackLengthSum->GetBinContent(i) / statTLE) - (meanL * meanL);
    // Is that not the wrong order? TODO: Ask.
    if (mean != 0.0 && N != 1 && squared != 0.0){
      //*po = sqrt( (1.0/(N-1))*(squared/N - pow(mean/N, 2)))/(mean/N);
      //*po = sqrt( ( 1.0 / (N-1) ) * ( squared/N - pow( mean/N , 2) ) ) / ( mean/N );
      *po = sqrt( pow( mean , 2 ) - squared );

    }
    else *po = 1;

    ++po;
    ++pi;
    ++pii;
  }
}


#endif /* end #define GATEIMAGEWITHSTATISTICTLE_HH */
