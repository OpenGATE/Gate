//
// Created by mdupont on 13/03/18.
//

#pragma once

#include <fstream>
#include <vector>
#include <functional>
#include <cxxabi.h>

#include <string>
#include <sstream>
#include <iostream>

#include "GateTreeFile.hh"
#include "GateTreeFileManager.hh"

class GateAsciiData;

typedef const std::function<void(const GateAsciiData *, std::fstream&)> save_to_file_f;
typedef const std::function<void(GateAsciiData *, const std::string&)> read_from_string_f;



class GateAsciiData : public GateData
{
public:
  GateAsciiData(const void *pointer_to_data, const std::string &name, const std::type_index &type_index,
            save_to_file_f &save_to_file, read_from_string_f &read_from_string);

public:
  save_to_file_f m_save_to_file;
  read_from_string_f m_read_from_string;
  size_t m_index_of_this_data_in_header;
  size_t m_max_caracter_accepted_by_provided_read_buffer;

};

template<typename T>
void save_to_file(const GateAsciiData *d, std::fstream &file)
{
  T *p = (T*)d->m_pointer_to_data;
  file << *p ;
}

template<typename T>
void read_from_string(GateAsciiData *d, const std::string& s)
{
    std::stringstream ss(s);

    T *p = (T*)d->m_pointer_to_data;
    ss >> *p ;
}

template<>
void read_from_string<char*>(GateAsciiData *d, const std::string& s);


class GateAsciiTree : public GateTree
{

public:
  GateAsciiTree();
  static std::string _get_factory_name() { return "txt"; }


 protected:
  void register_variable(const std::string &name, const void *p, std::type_index t_index) override;
  void register_variable(const std::string &name, const char *p, size_t nb_char) override;
  void register_variable(const std::string &name, const std::string *p, size_t nb_char) override;
  void register_variable(const std::string &name, const int *p, size_t n) override;

  template<typename T>
  void register_variable(const std::string &name, const T *p)
  {

      if(!std::is_arithmetic<T>::value)
      {
          throw std::invalid_argument(std::string("templated version of register_variable can not be used for type = ") +
                                      abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, nullptr));
      }
      register_variable(name, p, typeid(T));
  }

  std::unordered_map<std::type_index, save_to_file_f> m_tmapOfSave_to_file;
  std::unordered_map<std::type_index, read_from_string_f> m_tmapOfRead_from_string;


private:

  template<typename T>
  void add_save_to_file_function()
  {
      m_tmapOfSave_to_file.emplace(typeid(T), &save_to_file<T> );
  }

  template<typename T>
  void add_read_from_string_function()
  {
      m_tmapOfRead_from_string.emplace(typeid(T), &read_from_string<T> );
  }

  template<typename T>
  void attach_read_and_write_function()
  {
      add_read_from_string_function<T>();
      add_save_to_file_function<T>();
  }


 protected:
  std::vector<GateAsciiData> m_vector_of_pointer_to_data;
  std::string m_sep;
};


class GateOutputAsciiTreeFile: public GateAsciiTree, public GateOutputTreeFile
{
 public:
  GateOutputAsciiTreeFile() = default;
  void open(const std::string& s) override ;
  bool is_open() override ;
  void close() override ;


  void write_variable(const std::string &name, const void *p, std::type_index t_index) override;
  void write_variable(const std::string &name, const std::string *p, size_t nb_char)override ;
  void write_variable(const std::string &name, const char *p, size_t nb_char) override  ;
  void write_variable(const std::string &name, const int *p, size_t n) override  ;


  template<typename T >
  void write_variable(const std::string &name, const T *p)
  {
      register_variable(name, p);
  }

  void write_header() override;
  void write() override;
  void fill() override;

private:
  std::fstream m_file;
  static bool s_registered;
};


class GateInputAsciiTreeFile: public GateAsciiTree, public GateInputTreeFile {
 public:
  GateInputAsciiTreeFile();
  void open(const std::string& s) override ;
  void close() override ;
  bool is_open() override ;
  void read_header() override;
  void read_next_entrie() override;
  bool data_to_read() override ;
  uint64_t nb_elements() override ;

  void read_entrie(const uint64_t &i) override;


  void read_variable(const std::string &name, void *p, std::type_index t_index) override ;
  void read_variable(const std::string &name, std::string* p) override ;
  void read_variable(const std::string &name, char* p) override ; // dangerous !
  void read_variable(const std::string &name, char* p, size_t nb_char) override ;
  using GateInputTreeFile::read_variable;


  bool has_variable(const std::string &name) override;
  std::type_index get_type_of_variable(const std::string &name) override;

private:
  size_t  m_number_of_lines_in_file;
  size_t  m_nb_entries_read;
  std::vector<std::string> m_list_from_header;
  std::fstream m_file;
  bool m_read_header_called;
  size_t m_start_of_data;
  static bool s_registered;

};

