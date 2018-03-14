/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#ifndef GATEREGIONDOSESTAT_HH
#define GATEREGIONDOSESTAT_HH

#include "GateMiscFunctions.hh"
#include "GateImage.hh"
#include <memory>

//-----------------------------------------------------------------------------
class GateRegionDoseStat {

public:

  GateRegionDoseStat(int id);
  std::string ToString();
  void Update(long event_id, double edep, double density);

  typedef std::map<int, std::shared_ptr<GateRegionDoseStat>> IdToSingleRegionMapType;
  typedef std::map<int, std::vector<std::shared_ptr<GateRegionDoseStat>>> LabelToSeveralRegionsMapType;
  typedef std::map<int, std::vector<int>> IdToLabelsMapType;

  static void InitRegions(GateImageFloat & image,
                          IdToSingleRegionMapType & regionMap,
                          LabelToSeveralRegionsMapType & regionsMap);
  static void AddAggregatedRegion(IdToSingleRegionMapType & regionMap,
                                  LabelToSeveralRegionsMapType & regionsMap,
                                  IdToLabelsMapType & labelsMap);

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
