/*----------------------
   Copyright (C): OpenGATE Collaboration

This software is distributed under the terms
of the GNU Lesser General Public Licence (LGPL)
See LICENSE.md for further details
----------------------*/

#ifndef GateMultiPhotonTrajectoryNavigatorHelpers_h
#define GateMultiPhotonTrajectoryNavigatorHelpers_h

#include <unordered_map>
#include <unordered_set>
#include <vector>

/** Authors: Wojciech Krzemień, Mateusz Bała and Kamil Dulski
 *  Emails: wojciech.krzemien@ncbj.gov.pl, mateusz.bala@ncbj.gov.pl and kamil.dulski@gmail.com
 *  Organization: National Centre For Nuclear Research (NCBJ, https://ncbj.gov.pl), Poland
 *  Developed within the IMPET project: https://pet.ncbj.gov.pl/
 *  About: Pure helper algorithms for resolving ancestor and primary tracks from parent-track maps.
 **/

namespace MultiphotonTrajectoryResolver {

/**
 * @brief Resolves ancestor and primary track identifiers from parent links.
 *
 * This helper is intentionally independent from Geant4 trajectory classes so
 * that algorithmic behavior can be unit-tested with synthetic graphs.
 */
  /**
   * @brief Marks all visited tracks as unresolved in cache.
   *
   * Args:
   *   visited: Traversed track IDs.
   *   cache: Optional cache to update.
   */
  inline void clean_cache(const std::vector<int> &visited, std::unordered_map<int, int> *cache) {
    if (cache) {
      for (const int v : visited) {
        (*cache)[v] = 0;
      }
    }
  }

  /**
   * @brief Tries to resolve current track using cache and propagates result.
   *
   * Args:
   *   current: Current track ID being resolved.
   *   visited: Traversed track IDs.
   *   cache: Optional memoization cache.
   *   resolved: Output resolved track ID.
   *
   * Returns:
   *   True when cached resolution was found.
   */
  inline bool try_get_cached_resolution(
      int current,
      const std::vector<int> &visited,
      std::unordered_map<int, int> *cache,
      int *resolved) {
    if (!cache || !resolved) {
      return false;
    }

    const auto it_cache = cache->find(current);
    if (it_cache == cache->end()) {
      return false;
    }

    *resolved = it_cache->second;
    for (const int v : visited) {
      (*cache)[v] = *resolved;
    }
    return true;
  }

  /**
   * @brief Writes resolved track ID for current and visited nodes.
   *
   * Args:
   *   current: Current track ID.
   *   visited: Traversed track IDs.
   *   cache: Optional memoization cache.
   *   resolved: Resolved track ID to store.
   */
  inline void cache_resolution_for_current_and_visited(
      int current,
      const std::vector<int> &visited,
      std::unordered_map<int, int> *cache,
      int resolved) {
    if (!cache) {
      return;
    }

    (*cache)[current] = resolved;
    for (const int v : visited) {
      (*cache)[v] = resolved;
    }
  }

  /**
   * @brief Resolves the nearest ancestor belonging to a reference photon set.
   *
   * Args:
   *   track_id: Start track identifier to resolve.
   *   parent_by_track: Mapping track -> parent track.
   *   reference_photon_track_ids: Set of accepted ancestor photon tracks.
   *   cache: Optional memoization cache for resolved ancestors.
   *   max_steps: Optional traversal guard; <=0 enables auto-guard.
   *
   * Returns:
   *   Matching ancestor photon track ID, or 0 when unresolved.
   */
  inline int ResolveAncestorPhotonTrackID(
      int track_id,
      const std::unordered_map<int, int> &parent_by_track,
      const std::unordered_set<int> &reference_photon_track_ids,
      std::unordered_map<int, int> *cache,
      int max_steps = -1) {
    if (track_id <= 0) {
      return 0;
    }

    std::vector<int> visited;
    int current = track_id;
    int resolved = 0;
    const int guard = max_steps > 0 ? max_steps : static_cast<int>(parent_by_track.size()) + 2;

    for (int step = 0; step < guard; ++step) {
      if (try_get_cached_resolution(current, visited, cache, &resolved)) {
        return resolved;
      }

      if (reference_photon_track_ids.find(current) != reference_photon_track_ids.end()) {
        cache_resolution_for_current_and_visited(current, visited, cache, current);
        return current;
      }

      visited.push_back(current);

      const auto it_parent = parent_by_track.find(current);
      if (it_parent == parent_by_track.end()) {
        break;
      }

      current = it_parent->second;
      if (current == 0) {
        break;
      }
    }

    clean_cache(visited, cache);

    return 0;
  }

  /**
   * @brief Resolves the primary track ID for a given track.
   *
   * Args:
   *   track_id: Start track identifier to resolve.
   *   parent_by_track: Mapping track -> parent track.
   *   cache: Optional memoization cache for primary track IDs.
   *   max_steps: Optional traversal guard; <=0 enables auto-guard.
   *
   * Returns:
   *   Primary track ID, or 0 when unresolved.
   */
  inline int ResolvePrimaryTrackID(
      int track_id,
      const std::unordered_map<int, int> &parent_by_track,
      std::unordered_map<int, int> *cache,
      int max_steps = -1) {
    if (track_id <= 0) {
      return 0;
    }

    std::vector<int> visited;
    int current = track_id;
    int resolved = 0;
    const int guard = max_steps > 0 ? max_steps : static_cast<int>(parent_by_track.size()) + 2;

    for (int step = 0; step < guard; ++step) {
      if (try_get_cached_resolution(current, visited, cache, &resolved)) {
        return resolved;
      }

      visited.push_back(current);

      const auto it_parent = parent_by_track.find(current);
      if (it_parent == parent_by_track.end()) {
        break;
      }

      const int parent = it_parent->second;
      if (parent == 0) {
        cache_resolution_for_current_and_visited(current, visited, cache, current);
        return current;
      }

      current = parent;
    }

    clean_cache(visited, cache);

    return 0;
  }
}  // namespace MultiphotonTrajectoryResolver

#endif
