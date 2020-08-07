//
// Created by mdupont on 30/08/17.
//

#pragma once

#include <string>
#include <ostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>
#include <stdexcept>
#include <cxxabi.h>


#include "GateTreeFile.hh"



class GateNumpyData : public GateData
{
public:
  GateNumpyData(const void * pointer_to_data,
       const size_t size_of_data,
       const std::string _name,
       const std::string numpy_format,
       const std::type_index type_index

  ) : GateData(pointer_to_data, _name, type_index),
      m_size_of_data(size_of_data),
      m_numpy_format(numpy_format),
      m_nb_characters(0),
      m_type_index_read(type_index),
      buffer_read(0)
  {
    std::stringstream descr;
    descr << "('" << m_name << "', '" << numpy_format << "')";
    m_numpy_description = descr.str();
  }

  const size_t m_size_of_data;
  const std::string m_numpy_format;
  size_t m_nb_characters;
  std::string m_numpy_description;

  std::type_index m_type_index_read; // type index of read variable, in case where we want to read a string
  char *buffer_read ;
};

class GateNumpyTree : public GateTree
{
public:
  GateNumpyTree();
  static std::string _get_factory_name() { return "npy"; }
  uint64_t nb_elements();


protected:
  void register_variable(const std::string &name, const void *p, std::type_index t_index) override;
  void register_variable(const std::string &name, const char *p, size_t nb_char) override;
  void register_variable(const std::string &name, const std::string *p, size_t nb_char) override;
  void register_variable(const std::string &name, const int *, size_t n) override;

  template<typename T>
  void register_variable(const std::string &name, const T *p)
  {
    if(!std::is_arithmetic<T>::value)
    {
      throw std::invalid_argument(std::string("templated version of register_variable can not be used for type = ") +
                                      abi::__cxa_demangle(typeid(T).name(), 0, 0, nullptr));
    }
    register_variable(name, p, typeid(T));
  }

private:
  void register_variable(const std::string& name, const void *p, const size_t size, std::string numpy_format, std::type_index t_index);
  void fill_maps(std::type_index t_index, const std::string&);

  template <typename T>
  void fill_maps(const std::string s)
  {
      fill_maps(std::type_index(typeid(T)), s);
  }


protected:
  uint64_t m_nb_elements;
  std::vector<GateNumpyData> m_vector_of_pointer_to_data;
  std::fstream m_file;

  uint64_t m_position_before_shape;
  uint64_t m_position_after_shape;

  const std::string magic_prefix = "\x93NUMPY";
  std::unordered_map<std::type_index, std::string> m_tmap_cppToNumpy;
  std::unordered_map<std::string, std::type_index> m_tmap_numpyToCpp;

};


class GateOutputNumpyTreeFile: public GateNumpyTree, public GateOutputTreeFile
{
public:
  GateOutputNumpyTreeFile();
  void open(const std::string& s) override ;
  bool is_open() override;
  void close() override;

  void write_header() override ;
  void write() override ;
  virtual void fill() override;

  void write_variable(const std::string &name, const void *p, std::type_index t_index) override;
  void write_variable(const std::string &name, const std::string *p, size_t nb_char)override ;
  void write_variable(const std::string &name, const char *p, size_t nb_char) override  ;
   void write_variable(const std::string &name, const int  *p, size_t n) override  ;

  template<typename T >
  void write_variable(const std::string &name, const T *p)
  {
    register_variable(name, p);
  }

private:
  bool m_write_header_called;
  static bool s_registered;
};


class GateInputNumpyTreeFile: public GateNumpyTree, public GateInputTreeFile
{
public:
  GateInputNumpyTreeFile();

  void open(const std::string &name) override ;
  bool is_open() override;
  virtual void read_next_entrie() override ;
  void close() override;
  uint64_t nb_elements() override ;

  void read_entrie(const uint64_t &i) override;

  void read_variable(const std::string &name, void *p, std::type_index t_index) override ;
  void read_variable(const std::string &name, char* p) override ;
  void read_variable(const std::string &name, std::string *p) override ;
  using GateInputTreeFile::read_variable; //call templated version
  void read_header() override;
  bool data_to_read() override ;

  bool has_variable(const std::string &name) override;
  std::type_index get_type_of_variable(const std::string &name) override;

private:
  size_t  m_length_of_file;
  static bool s_registered;
  bool m_read_header_called;
  size_t m_start_of_data;
};

