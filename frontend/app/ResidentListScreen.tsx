import React, { useState, useMemo } from "react";
import {
  View,
  Text,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  TextInput,
  Animated,
  Dimensions,
} from "react-native";
import { colors } from "../theme/colors";
import RiskBadge from "./components/RiskBadge";
import SectionHeader from "./components/SectionHeader";
import StatsBar from "./components/StatsBar";

const { width } = Dimensions.get("window");

const mockResidents = [
  {
    id: "1",
    name: "Margaret Johnson",
    age: 78,
    alerts: 5,
    risk: "High",
    birthMonth: 11,
    birthDay: 24,
  },
  {
    id: "2",
    name: "Robert Miller",
    age: 82,
    alerts: 3,
    risk: "Medium",
    birthMonth: 3,
    birthDay: 15,
  },
  {
    id: "3",
    name: "Helen Davis",
    age: 75,
    alerts: 2,
    risk: "Low",
    birthMonth: 7,
    birthDay: 22,
  },
  {
    id: "4",
    name: "Eleanor White",
    age: 88,
    alerts: 4,
    risk: "High",
    birthMonth: 12,
    birthDay: 10,
  },
  {
    id: "5",
    name: "William Brown",
    age: 80,
    alerts: 1,
    risk: "Low",
    birthMonth: 6,
    birthDay: 8,
  },
];

export default function ResidentListScreen({ navigation }) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState("risk"); // "risk" or "name"
  const [filterType, setFilterType] = useState(null); // null, "highRisk", or "alerts"

  const stats = [
    { value: 42, label: "Total Residents" },
    { value: 8, label: "High Risk" },
    { value: 15, label: "Total Alerts" },
  ];

  // Sort residents by risk level or name
  const sortedResidents = useMemo(() => {
    let sorted = [...mockResidents];

    // Apply filter first
    if (filterType === "highRisk") {
      sorted = sorted.filter((resident) => resident.risk === "High");
    } else if (filterType === "alerts") {
      sorted = sorted.filter((resident) => resident.alerts > 0);
      sorted.sort((a, b) => b.alerts - a.alerts); // Sort by most alerts
    }

    if (sortBy === "risk") {
      const riskOrder = { High: 0, Medium: 1, Low: 2 };
      sorted.sort((a, b) => riskOrder[a.risk] - riskOrder[b.risk]);
    } else {
      sorted.sort((a, b) => a.name.localeCompare(b.name));
    }

    // Filter by search
    if (search.trim()) {
      sorted = sorted.filter((resident) =>
        resident.name.toLowerCase().includes(search.toLowerCase())
      );
    }

    return sorted;
  }, [search, sortBy, filterType]);

  // Get initials for avatar
  const getInitials = (name) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase();
  };

  // Get background color based on risk level
  const getRiskBackgroundColor = (risk) => {
    switch (risk) {
      case "High":
        return "#FFE5E5";
      case "Medium":
        return "#FFF8E5";
      case "Low":
        return "#E5F5F0";
      default:
        return colors.alertCard;
    }
  };

  // Get border color based on risk level
  const getRiskBorderColor = (risk) => {
    switch (risk) {
      case "High":
        return "#FF6B6B";
      case "Medium":
        return "#FFA500";
      case "Low":
        return "#2ECC71";
      default:
        return colors.green;
    }
  };

  // Handle stat clicks
  const handleStatPress = (index) => {
    if (index === 0) {
      setFilterType(null);
      setSearch("");
    } else if (index === 1) {
      setFilterType("highRisk");
    } else if (index === 2) {
      setFilterType("alerts");
    }
  };

  // Get filter title
  const getFilterTitle = () => {
    switch (filterType) {
      case "highRisk":
        return "High Risk Residents";
      case "alerts":
        return "Residents with Alerts";
      default:
        return "Residents";
    }
  };

  // Check if today is the resident's birthday
  const isBirthday = (birthMonth, birthDay) => {
    const today = new Date();
    return today.getMonth() + 1 === birthMonth && today.getDate() === birthDay;
  };

  return (
    <View style={styles.container}>
      <SectionHeader
        title={getFilterTitle()}
        subtitle={filterType ? "Tap to view all residents" : "View resident profiles"}
      />
      <StatsBar stats={stats} onStatPress={handleStatPress} />

      {/* Search Bar */}
      <View style={styles.searchContainer}>
        <TextInput
          style={styles.searchInput}
          placeholder="Search residents..."
          placeholderTextColor="#999"
          value={search}
          onChangeText={setSearch}
        />
      </View>

      {/* Active Filter Indicator */}
      {filterType && (
        <TouchableOpacity
          style={styles.filterBanner}
          onPress={() => {
            setFilterType(null);
            setSearch("");
          }}
        >
          <Text style={styles.filterBannerText}>
            ✕ Clear filter ({sortedResidents.length} results)
          </Text>
        </TouchableOpacity>
      )}

      {/* Sort Buttons */}
      <View style={styles.sortContainer}>
        <TouchableOpacity
          style={[
            styles.sortButton,
            sortBy === "risk" && styles.sortButtonActive,
          ]}
          onPress={() => setSortBy("risk")}
        >
          <Text
            style={[
              styles.sortButtonText,
              sortBy === "risk" && styles.sortButtonTextActive,
            ]}
          >
            By Risk
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[
            styles.sortButton,
            sortBy === "name" && styles.sortButtonActive,
          ]}
          onPress={() => setSortBy("name")}
        >
          <Text
            style={[
              styles.sortButtonText,
              sortBy === "name" && styles.sortButtonTextActive,
            ]}
          >
            By Name
          </Text>
        </TouchableOpacity>
      </View>

      {sortedResidents.length > 0 ? (
        <FlatList
          data={sortedResidents}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={[
                styles.residentCard,
                {
                  backgroundColor: getRiskBackgroundColor(item.risk),
                  borderLeftColor: getRiskBorderColor(item.risk),
                },
              ]}
              activeOpacity={0.7}
              onPress={() =>
                navigation.navigate("ResidentProfile", { resident: item })
              }
            >
              {/* Avatar */}
              <View style={styles.avatar}>
                <Text style={styles.avatarText}>{getInitials(item.name)}</Text>
              </View>

              {/* Card Content */}
              <View style={styles.cardContent}>
                <View style={styles.nameSection}>
                  <Text style={styles.name}>{item.name}</Text>
                  <Text style={styles.age}>
                    {isBirthday(item.birthMonth, item.birthDay) ? "🎂" : "👤"} Age{" "}
                    {item.age}
                    {isBirthday(item.birthMonth, item.birthDay) &&
                      " - Extra vigilance today!"}
                  </Text>
                </View>
                <View style={styles.statsSection}>
                  <RiskBadge risk={item.risk} />
                  <View style={styles.alertBadge}>
                    <Text style={styles.alertIcon}>🔔</Text>
                    <Text style={styles.alertCount}>{item.alerts}</Text>
                  </View>
                </View>
              </View>
            </TouchableOpacity>
          )}
        />
      ) : (
        <View style={styles.emptyState}>
          <Text style={styles.emptyStateText}>No residents found</Text>
          <Text style={styles.emptyStateSubtext}>Try a different search term</Text>
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.white,
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  searchContainer: {
    marginBottom: 16,
    marginTop: 12,
  },
  searchInput: {
    backgroundColor: "#F5F5F5",
    borderRadius: 12,
    padding: 12,
    fontSize: 15,
    color: colors.textDark,
    borderWidth: 1,
    borderColor: "#E0E0E0",
  },
  filterBanner: {
    backgroundColor: colors.green,
    borderRadius: 10,
    padding: 12,
    marginBottom: 12,
    flexDirection: "row",
    alignItems: "center",
  },
  filterBannerText: {
    color: colors.textDark,
    fontWeight: "600",
    fontSize: 14,
  },
  sortContainer: {
    flexDirection: "row",
    marginBottom: 16,
    gap: 10,
  },
  sortButton: {
    flex: 1,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: "#F0F0F0",
    borderWidth: 1,
    borderColor: "#E0E0E0",
    alignItems: "center",
  },
  sortButtonActive: {
    backgroundColor: colors.green,
    borderColor: colors.lightGreen,
  },
  sortButtonText: {
    fontSize: 14,
    fontWeight: "600",
    color: "#666",
  },
  sortButtonTextActive: {
    color: colors.textDark,
  },
  residentCard: {
    backgroundColor: colors.alertCard,
    borderRadius: 14,
    padding: 14,
    marginBottom: 12,
    borderLeftWidth: 5,
    borderLeftColor: colors.green,
    flexDirection: "row",
    alignItems: "center",
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.08,
    shadowRadius: 6,
    elevation: 3,
  },
  avatar: {
    width: 48,
    height: 48,
    borderRadius: 24,
    backgroundColor: colors.pink,
    justifyContent: "center",
    alignItems: "center",
    marginRight: 12,
  },
  avatarText: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
  },
  cardContent: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    flex: 1,
  },
  nameSection: {
    flex: 1,
  },
  name: {
    fontSize: 15,
    fontWeight: "700",
    color: colors.textDark,
    marginBottom: 4,
  },
  age: {
    fontSize: 13,
    color: colors.textSecondary,
    fontWeight: "500",
  },
  statsSection: {
    alignItems: "flex-end",
    gap: 8,
    marginLeft: 10,
  },
  alertBadge: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#FFF3CD",
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: 8,
    gap: 4,
  },
  alertIcon: {
    fontSize: 12,
  },
  alertCount: {
    fontSize: 12,
    fontWeight: "600",
    color: "#996600",
  },
  emptyState: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    paddingBottom: 100,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: "600",
    color: colors.textDark,
    marginBottom: 8,
  },
  emptyStateSubtext: {
    fontSize: 14,
    color: colors.textSecondary,
  },
});
