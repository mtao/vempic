#include "vem/serialization/inventory.hpp"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>

namespace {

const std::string current_inventory_version = "0.0.1";

std::string get_unique_name(
    const std::string& name, const std::string& extension,
    const std::function<bool(const std::filesystem::path&)>& predicate) {
    std::string guess;
    if (extension.empty()) {
        guess = name;
    } else {
        guess = name + "." + extension;
    }
    if (predicate(guess)) {
        return guess;
    }
    // infinite loop... famous last words "it'll never happen"
    std::string format;
    if (extension.empty()) {
        format = name + "-{}";
    } else {
        format = name + "-{}." + extension;
    }
    int try_num = 0;
    do {
        guess = fmt::vformat(format, fmt::make_format_args(try_num++));
    } while (!predicate(guess));
    return guess;
}
std::string get_unique_path(const std::string& name,
                            const std::string& extension) {
    return get_unique_name(
        name, extension, [](auto&& p) { return !std::filesystem::exists(p); });
    // std::bind(&std::filesystem::exists, std::placeholders::_1));
}

}  // namespace
namespace vem::serialization {

    Inventory Inventory::from_options(const cxxopts::ParseResult& result) {

    char* path = std::getenv("SCRATCH_DIRECTORY");
    std::string inventory_name = result["inventory_name"].as<std::string>();
    std::filesystem::path root_dir;
    if (path == nullptr) {
        root_dir = std::filesystem::path(".") / inventory_name;
    } else {
        root_dir =
            std::filesystem::absolute(path) / "unorganized" / inventory_name;
    }
    root_dir = get_unique_path(root_dir, {});

    std::filesystem::create_directory(root_dir);
    bool force_new = result["force_overwrite"].count() == 0;
    return root_dir;
    }

    cxxopts::OptionAdder& Inventory::add_options(cxxopts::Options& opts) {

        return opts.add_options("inventory")
            ("force_overwrite", "overwrite an inventory with the same name")
            ("inventory_name", "name of the domain", cxxopts::value<std::string>());

    }
Inventory::Inventory(Inventory&& o) {
    _child_inventories = std::move(o._child_inventories);
    _dirty = o._dirty;
    _immediate_mode = o._immediate_mode;
    _js = std::move(o._js);
    _path = std::move(o._path);
    _root = o._root;
    _subinventory_storage_dirty = o._subinventory_storage_dirty;
    o.clear();
}
Inventory& Inventory::operator=(Inventory&& o) {
    save();

    _child_inventories = std::move(o._child_inventories);
    _dirty = o._dirty;
    _immediate_mode = o._immediate_mode;
    _js = std::move(o._js);
    _path = std::move(o._path);
    _root = o._root;
    _subinventory_storage_dirty = o._subinventory_storage_dirty;
    o.clear();
    return *this;
}

void Inventory::clear() {
    _js = nullptr;
    _root = nullptr;
    // path shouldn't really matter eh?
    _dirty = false;
    _subinventory_storage_dirty = false;
    _immediate_mode = false;
}

void Inventory::set_dirty() const {}
void Inventory::set_dirty() {
    _dirty = true;
    if (_immediate_mode) {
        save();
    }
}
void Inventory::set_subinventory_dirty() const {}
void Inventory::set_subinventory_dirty() {
    _subinventory_storage_dirty = true;
    if (_immediate_mode) {
        save();
    }
}
Inventory Inventory::from_scratch(const std::string& inventory_name,
                                  bool force_new) {
    char* path = std::getenv("SCRATCH_DIRECTORY");
    std::filesystem::path root_dir;
    if (path == nullptr) {
        root_dir = std::filesystem::path(".") / inventory_name;
    } else {
        root_dir =
            std::filesystem::absolute(path) / "unorganized" / inventory_name;
    }
    root_dir = get_unique_path(root_dir, {});

    std::filesystem::create_directory(root_dir);
    return root_dir;
}
Inventory::~Inventory() { save(); }
std::filesystem::path Inventory::real_path() const {
    if (_root == nullptr) {
        return _path;
    } else {
        return _root->real_path() / _path;
    }
}

void Inventory::save() {
    if (_dirty && !_js.is_null()) {
        std::ofstream(inventory_path()) << _js;
    }
    if (_subinventory_storage_dirty && !_child_inventories.is_null()) {
        std::ofstream(subinventory_path()) << _child_inventories;
    }
}

void Inventory::reload() {
    std::filesystem::path inventory_path = this->inventory_path();
    bool inventory_existed = std::filesystem::is_regular_file(inventory_path);

    if (inventory_existed) {
        try {
            std::ifstream ifs(inventory_path);
            ifs >> _js;
        } catch (std::exception& e) {
            spdlog::warn("Inventory reload got an error: {}", e.what());
        }
    }
    std::filesystem::path subinventory_path = this->subinventory_path();
    bool subinventory_existed =
        std::filesystem::is_regular_file(subinventory_path);
    if (subinventory_existed) {
        try {
            std::ifstream ifs(subinventory_path);
            ifs >> _child_inventories;
        } catch (std::exception& e) {
            spdlog::warn("Inventory reload got an error: {}", e.what());
        }
    }
}

Inventory::Inventory(const std::filesystem::path& path, const Inventory* root,
                     bool must_exist, bool assert_on_failure)
    : _path(path), _root(root) {
    if (std::filesystem::is_regular_file(path) &&
        path.filename() == ".inventory.json") {
        _path = path.parent_path();
    }
    std::filesystem::path inventory_path = this->inventory_path();
    bool inventory_existed = std::filesystem::is_regular_file(inventory_path);

    if (inventory_existed) {
        try {
            std::ifstream(inventory_path) >> _js;
        } catch (const std::exception& e) {
            spdlog::info("Failed to open inventory {}: {}",
                         std::string(inventory_path), e.what());
            if (assert_on_failure) {
                throw e;
            }
        }

        auto p = subinventory_path();
        if (std::filesystem::is_regular_file(p)) {
            try {
                std::ifstream(p) >> _child_inventories;
            } catch (const std::exception& e) {
                spdlog::info("Failed to open subinventory {}: {}",
                             std::string(p), e.what());

                if (assert_on_failure) {
                    throw e;
                }
            }
        }
    } else if (auto old_inventory_path = real_path() / "inventory.json";
               std::filesystem::is_regular_file(old_inventory_path)) {
        spdlog::info("Found an older-style inventory. loading it");

        try {
            std::ifstream ifs(old_inventory_path);
            ifs >> _js;

            update_inventory_storage();
            inventory_existed = true;

        } catch (std::exception& e) {
            spdlog::warn("Loading an old inventory got an error: {}", e.what());
            if (assert_on_failure) {
                throw e;
            }
        }
    } else if (!must_exist) {
        _js["version"] = current_inventory_version;
        _dirty = true;
    } else {
        throw std::runtime_error(
            fmt::format("Inventory not found [{}]", std::string(_path)));
    }
}
Inventory Inventory::make_subinventory(const std::string& name) {
    std::filesystem::path p = _path / name;
    std::filesystem::create_directory(p);
    _child_inventories[name]["name"] = std::string(name);
    set_subinventory_dirty();
    Inventory ret(p);
    if (_immediate_mode) {
        ret.set_immediate_mode();
    }
    return ret;
}
const Inventory Inventory::get_subinventory(const std::string& name) const {
    std::filesystem::path p = _path / name;
    if (std::filesystem::is_directory(p)) {
        // get an inventory that will assert on failure
        return Inventory(p, nullptr, true, true);
    }

    throw std::runtime_error(fmt::format(
        "SubInventory not found because directory did not exist [{}]",
        std::string(_path)));
}

bool Inventory::has_subinventory(const std::string& name) const {
    return _child_inventories.contains(name);
}
std::vector<std::string> Inventory::get_subinventory_names() const {
    std::vector<std::string> ret;
    for (auto&& [name, meta] : _child_inventories.items()) {
        ret.emplace_back(name);
        /*
    const auto& namem = meta.at["name"];
    if (namem.is_string()) {
        ret.emplace_back(namem.get<std::string>());
    } else {
        spdlog::info(
            "Child inventory name was not a string, but rather a {}",
            std::string(namem));
    }
    */
    }
    return ret;
}

// returns a path to a file in the inventory if it exists
std::filesystem::path Inventory::get_asset_path(
    const std::string& asset_name) const {
    return _path / _js["assets"][asset_name]["path"];
    /*
if (_js["assets"].contains(asset_name)) {
    return _path / _js["assets"][asset_name]["path"];
} else {
    return {};
}
*/
}
void Inventory::move_old_metadata(const std::string& key) {
    if (_js.contains(key)) {
        if (!_js["overwritten_metadata"].contains(key)) {
            _js["overwritten_metadata"][key] = nlohmann::json::array();
        }
        _js["overwritten_metadata"][key].push_back(_js[key]);
    }
}

// returns a path for a new asset, user is responsible for writing to the path
std::filesystem::path Inventory::get_new_asset_path(
    const std::string& asset_name, const std::string& extension,
    bool is_directory, bool touch_file, bool create_new_path) {
    if (_js["assets"].contains(asset_name)) {
        spdlog::warn("Asset [{}] is being overwritten", asset_name);
        if (!_js["overwritten_assets"].contains(asset_name)) {
            _js["overwritten_assets"][asset_name] = nlohmann::json::array();
        }
        auto& arr = _js["overwritten_assets"][asset_name];
        arr.push_back(_js["assets"][asset_name]);
    }

    auto& obj = _js["assets"][asset_name] = nlohmann::json::object();
    std::filesystem::path new_path;
    if (create_new_path) {
        new_path = get_unique_path(real_path() / asset_name, extension);
        if (touch_file) {
            if (is_directory) {
                std::filesystem::create_directory(new_path);
            } else {
                std::ofstream ofs(new_path);
            }
        }
    } else {
        if (extension.empty()) {
            new_path = real_path() / asset_name;
        } else {
            new_path = real_path() / (asset_name + "." + extension);
        }
    }
    obj["path"] = std::filesystem::relative(new_path, real_path());
    set_dirty();

    return new_path;
}

nlohmann::json& Inventory::asset_metadata(const std::string& asset_name) {
    return _js["assets"].at(asset_name);
}
const nlohmann::json& Inventory::asset_metadata(
    const std::string& asset_name) const {
    return _js["assets"].at(asset_name);
}

std::filesystem::path Inventory::inventory_path() const {
    return real_path() / ".inventory.json";
}
std::filesystem::path Inventory::subinventory_path() const {
    return real_path() / ".child_inventories.json";
}

void Inventory::update_inventory_storage() {
    if (_js.contains(
            "child_inventories")) {  // oldest version, no version tag yet!
        _child_inventories = _js["child_inventories"];
        _js.erase("child_inventories");
        _js["version"] = current_inventory_version;  // we've up
        _dirty = true;
        _subinventory_storage_dirty = true;
        return;
    } else if (!_js.contains("version")) {
        spdlog::debug("Version not found!");
        return;
    }
    std::string version_tag = _js["version"];
    if (version_tag == "0.0.0") {  // experiment
    }
}

void Inventory::copy_asset_from_subinventory(const std::string& subinv_name,
                                             const std::string& asset_name) {
    copy_asset_from_inventory(get_subinventory(subinv_name), asset_name);
}
void Inventory::copy_asset_from_inventory(const Inventory& other,
                                          const std::string& asset_name) {
    // welp

    try {
        auto& o_meta = other.asset_metadata(asset_name);
        const std::string o_filename = o_meta["path"];
        size_t last_dot = o_filename.find_last_of('.') + 1;

        std::string extension;
        if (last_dot < o_filename.size()) {
            extension = o_filename.substr(last_dot);
        }

        auto opath = other.get_asset_path(asset_name);
        auto path = get_new_asset_path(asset_name, extension);
        auto& asset_meta = asset_metadata(asset_name);
        asset_meta = o_meta;
        asset_meta["path"] = path;
        // remove the test file initially created to reserve teh asset
        std::filesystem::remove(path);
        std::filesystem::create_hard_link(opath, path);
    } catch (const nlohmann::detail::out_of_range& e) {
        spdlog::info(
            "Copying asset from inventory failed because the other inventory "
            "didnt have the desired asset ({} not found among [{}])",
            asset_name, fmt::join(other.asset_names(), ","));
    }
}

std::vector<std::string> Inventory::asset_names() const {
    std::vector<std::string> ret;
    for (auto&& [name, meta] : _js["assets"].items()) {
        ret.emplace_back(name);
    }
    return ret;
}
}  // namespace vem::serialization
