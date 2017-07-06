/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See GATE/LICENSE.txt for further details
  ----------------------*/

#ifndef GATEREGIONDOSESTAT_HH
#define GATEREGIONDOSESTAT_HH

#include "GateMiscFunctions.hh"
#include "GateImage.hh"

//-----------------------------------------------------------------------------
class GateRegionDoseStat {

public:

  GateRegionDoseStat();
  std::string ToString();
  void Update(long event_id, double edep, double density);

  static void ComputeRegionVolumes(GateImageInt & image,
                                   std::map<int, GateRegionDoseStat> & mMapOfRegionStat);

  double sum_edep;
  double sum_squared_edep;
  double sum_temp_edep;
  double sum_dose;
  double sum_squared_dose;
  double sum_temp_dose;
  double volume;
  long last_event_id;
  long nb_hits;
  long nb_event_hits;
};
//-----------------------------------------------------------------------------

#endif /* end #define GATEREGIONDOSESTAT_HH */
