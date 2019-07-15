//
// Created by mdupont on 17/12/18.
//

#include "GateFile.hh"


void GateFile::open(const std::string&s , std::ios_base::openmode mode)
{
//  cout << "TreeFile::open with mode = " << mode << endl;

  m_mode = mode;
  m_path = s;
}

const std::string &GateFile::path() const
{
  return m_path;
}


