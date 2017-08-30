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

  GateRegionDoseStat(int id);
  std::string ToString();
  void Update(long event_id, double edep, double density);

  typedef std::map<int, std::shared_ptr<GateRegionDoseStat>> LabelToSingleRegionMapType;
  typedef std::map<int, std::vector<std::shared_ptr<GateRegionDoseStat>>> LabelToSeveralRegionsMapType;

  static void ComputeRegionVolumes(GateImageFloat & image,
                                   LabelToSingleRegionMapType & mMapOfRegionStat);
  static void AddAggregatedRegion(LabelToSingleRegionMapType & map,
                                  LabelToSeveralRegionsMapType & regionsMap,
                                  std::vector<int> & labels);

  int id;
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
