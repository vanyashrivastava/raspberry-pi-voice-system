import React from "react";
import { View, Text, StyleSheet, FlatList, TouchableOpacity } from "react-native";
import { colors } from "../theme/colors";
import RiskBadge from "./components/RiskBadge";
import SectionHeader from "./components/SectionHeader";
import StatsBar from "./components/StatsBar";

const mockResidents = [
  {
    id: "1",
    name: "Margaret Johnson",
    age: 78,
    alerts: 5,
    risk: "High",
  },
  {
    id: "2",
    name: "Robert Miller",
    age: 82,
    alerts: 3,
    risk: "Medium",
  },
  {
    id: "3",
    name: "Helen Davis",
    age: 75,
    alerts: 2,
    risk: "Low",
  },
  {
    id: "4",
    name: "Eleanor White",
    age: 88,
    alerts: 4,
    risk: "High",
  },
  {
    id: "5",
    name: "William Brown",
    age: 80,
    alerts: 1,
    risk: "Low",
  },
];

export default function ResidentListScreen({ navigation }) {
  const stats = [
    { value: 42, label: "Total Residents" },
    { value: 8, label: "High Risk" },
    { value: 15, label: "Total Alerts" },
  ];

  return (
    <View style={styles.container}>
      <SectionHeader
        title="Residents"
        subtitle="View and manage resident profiles"
      />
      <StatsBar stats={stats} />

      <FlatList
        data={mockResidents}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.residentCard}
            onPress={() => navigation.navigate("ResidentProfile", { resident: item })}
          >
            <View style={styles.cardContent}>
              <View style={styles.nameSection}>
                <Text style={styles.name}>{item.name}</Text>
                <Text style={styles.age}>Age {item.age}</Text>
              </View>
              <View style={styles.statsSection}>
                <RiskBadge risk={item.risk} />
                <Text style={styles.alertCount}>{item.alerts} alerts</Text>
              </View>
            </View>
          </TouchableOpacity>
        )}
      />
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
  residentCard: {
    backgroundColor: colors.alertCard,
    borderRadius: 14,
    padding: 16,
    marginBottom: 12,
    borderLeftWidth: 4,
    borderLeftColor: colors.green,
  },
  cardContent: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  nameSection: {
    flex: 1,
  },
  name: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
    marginBottom: 4,
  },
  age: {
    fontSize: 13,
    color: "#666",
  },
  statsSection: {
    alignItems: "flex-end",
    gap: 8,
  },
  alertCount: {
    fontSize: 13,
    color: "#999",
  },
});
