#pragma once
#include <filesystem>
#include <nlohmann/json.hpp>
#include <cxxopts.hpp>

namespace vem::serialization {
class Inventory {
   public:
    // creates a new inventory at the environment variable SCRATCH_DIRECTORY or
    // . if not found
    static Inventory from_scratch(const std::string& inventory_name,
                                  bool force_new = true);
    static Inventory from_options(const cxxopts::ParseResult& pr);

    static cxxopts::OptionAdder& add_options(cxxopts::Options& opts);

   protected:
    // should only used to cast a const Inventory to a derived type
    Inventory(const Inventory&) = default;

   public:
    Inventory(Inventory&&);
    Inventory& operator=(Inventory&&);
    Inventory(const std::filesystem::path& path,
              const Inventory* root = nullptr, bool must_exist = false,
              bool throw_if_doesnt_exist = false);
    ~Inventory();

    void set_immediate_mode(bool y = true) { _immediate_mode = y; }
    void unset_immediate_mode() { _immediate_mode = false; }

    // returns a path to a file in the inventory if it exists
    std::filesystem::path get_asset_path(const std::string& asset_name) const;

    const nlohmann::json& metadata() const { return _js; }
    const nlohmann::json& metadata(const std::string& key) const {
        return _js.at(key);
    }

    nlohmann::json& metadata() { return _js; }
    nlohmann::json& metadata(const std::string& key) { return _js[key]; }

    nlohmann::json& asset_metadata(const std::string& asset_name);
    const nlohmann::json& asset_metadata(const std::string& asset_name) const;
    std::vector<std::string> asset_names() const;

    // constness means the mutability of the database
    void save();
    void reload();

    // returns a path for a new asset, user is responsible for writing to the
    // path, but an empty file or directory is placed to prevent asynchrony
    // issues
    std::filesystem::path get_new_asset_path(
        const std::string& asset_name, const std::string& extension = "dat",
        bool is_directory = false, bool touch_file = true,
        bool create_new_path = true);

    // creates a new inventory inside of this inventory's structure
    Inventory make_subinventory(const std::string& name);
    // returns an immutable inventory
    const Inventory get_subinventory(const std::string& name) const;
    std::vector<std::string> get_subinventory_names() const;
    bool has_subinventory(const std::string& name) const;

    template <typename T>
    void add_metadata(const std::string& key, T&& value,
                      bool archive_old = true);
    template <typename T>
    void add_metadata(const std::string& key,
                      const std::initializer_list<T>& value,
                      bool archive_old = true);

    std::filesystem::path real_path() const;

    std::filesystem::path inventory_path() const;
    std::filesystem::path subinventory_path() const;

    void copy_asset_from_subinventory(const std::string& subinv_name, const std::string& asset_name);
    void copy_asset_from_inventory(const Inventory& other, const std::string& asset_name);

   private:
    void set_dirty();
    void set_dirty() const;
    void set_subinventory_dirty();
    void set_subinventory_dirty() const;
    // clean things up so we wont write anything at all
    void clear();
    nlohmann::json _js;
    nlohmann::json _child_inventories;
    const Inventory* _root = nullptr;
    std::filesystem::path _path;
    bool _dirty = false;
    bool _subinventory_storage_dirty = false;
    // save instantly instead of worrying about hte dirty flag
    bool _immediate_mode = true;

    void update_inventory_storage();
    void move_old_metadata(const std::string& key);
};
template <typename T>
void Inventory::add_metadata(const std::string& key, T&& value,
                             bool archive_old) {
    if (archive_old) move_old_metadata(key);
    _js[key] = value;
    set_dirty();
}
template <typename T>
void Inventory::add_metadata(const std::string& key,
                             const std::initializer_list<T>& value,
                             bool archive_old) {
    if (archive_old) move_old_metadata(key);
    _js[key] = value;
    set_dirty();
}
}  // namespace vem::serialization
