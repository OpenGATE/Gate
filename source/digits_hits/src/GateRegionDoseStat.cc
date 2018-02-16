/*----------------------
  Copyright (C): OpenGATE Collaboration

  This software is distributed under the terms
  of the GNU Lesser General  Public Licence (LGPL)
  See LICENSE.md for further details
  ----------------------*/

#include "GateRegionDoseStat.hh"

//-----------------------------------------------------------------------------
GateRegionDoseStat::GateRegionDoseStat(int i):id(i)
{
  sum_edep = 0.0;
  sum_squared_edep = 0.0;
  sum_temp_edep = 0.0;
  sum_dose = 0.0;
  sum_squared_dose = 0.0;
  sum_temp_dose = 0.0;
  last_event_id = -1;
  nb_hits = 0;
  nb_event_hits = 0;
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
std::string GateRegionDoseStat::ToString()
{
  std::ostringstream oss;
  oss << id << " "
      << sum_edep << " " << sum_squared_edep << " " << sum_temp_edep
      << sum_dose << " " << sum_squared_dose << " " << sum_temp_dose
      << " last_event=" << last_event_id << " nb_hits=" << nb_hits
      << " nb_e_hits = " << nb_event_hits << " vol=" << volume;
  return oss.str();
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
void GateRegionDoseStat::Update(long event_id, double edep, double density)
{
  double dose = edep/density/volume/gray;
  if (edep == 0.0) dose = 0.0;
  ++nb_hits;
  if (event_id != last_event_id) {
    sum_squared_edep += (sum_temp_edep*sum_temp_edep);
    sum_edep += sum_temp_edep;
    sum_temp_edep = edep;
    sum_squared_dose += (sum_temp_dose*sum_temp_dose);
    sum_dose += sum_temp_dose;
    sum_temp_dose = dose;
    last_event_id = event_id;
    ++nb_event_hits;
  }
  else {
    sum_temp_edep += edep;
    sum_temp_dose += dose;
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Static
void GateRegionDoseStat::InitRegions(GateImageFloat & image,
                                     IdToSingleRegionMapType & regionMap,
                                     LabelToSeveralRegionsMapType & regionsMap)
{
  GateImageFloat::const_iterator pi = image.begin();
  while (pi != image.end()) {
    int label = *pi;
    auto it = regionMap.find(label);
    if (it == regionMap.end()) {
      auto region = std::make_shared<GateRegionDoseStat>(label);
      regionMap[label] = region;
    }
    regionMap[label]->volume += 1.0;
    ++pi;
  }
  for(auto &m:regionMap) {
    m.second->volume *= image.GetVoxelVolume();
    regionsMap[m.first].push_back(m.second);
  }
}
//-----------------------------------------------------------------------------


//-----------------------------------------------------------------------------
// Static
void GateRegionDoseStat::AddAggregatedRegion(IdToSingleRegionMapType & regionMap,
                                             LabelToSeveralRegionsMapType & regionsMap,
                                             IdToLabelsMapType & labelsMap)
{
  for (auto labels:labelsMap) {
    // Check if the label for the new region is already used
    for (auto regions:regionsMap) {
      for (auto region:regions.second) {
        if (labels.first == region->id)
          throw std::runtime_error("[GATE] the label "+std::to_string(labels.first)+" for the new region already exist.");
      }
    }
    // Create region
    auto region = std::make_shared<GateRegionDoseStat>(labels.first);
    region->volume = 0;
    // Update region and set in the map
    for (auto label:labels.second) {
      region->volume += regionMap[label]->volume;
      regionsMap[label].push_back(region);
    }
    // Also add the new aggregated region to the map of single regions
    regionMap[labels.first] = region;
  }
}
//-----------------------------------------------------------------------------

