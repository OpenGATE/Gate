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

#include "GateImageWithStatistic.hh"

//-----------------------------------------------------------------------------
/// \brief
class GateImageWithStatisticTLE : public GateImageWithStatistic
{
public:
  virtual void UpdateUncertaintyImage(int numberOfEvents);

};

#endif /* end #define GATEIMAGEWITHSTATISTICTLE_HH */
