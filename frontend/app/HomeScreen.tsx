import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, FlatList, ScrollView } from "react-native";
import { colors } from "../theme/colors";

export default function HomeScreen({ navigation }) {
  const nursingHomeName = "Silver Oaks Nursing Home";
  const residentsMonitored = 42;

  const scamAlerts = [
    { id: "1", resident: "Evelyn Carter", risk: "High", type: "Bank phishing text" },
    { id: "2", resident: "Howard Miles", risk: "Medium", type: "Medicare scam call" },
  ];

  const menuItems = [
    { label: "📧 Email Alerts", screen: "EmailAlerts" },
    { label: "📞 Call Monitoring", screen: "CallMonitoring" },
    { label: "👥 Residents", screen: "ResidentList" },
    { label: "⚙️ Settings", screen: "Settings" },
  ];

  return (
    <ScrollView style={styles.container}>
      <Text style={styles.header}>{nursingHomeName}</Text>
      <Text style={styles.subHeader}>
        Monitoring {residentsMonitored} residents
      </Text>

      <View style={styles.menuGrid}>
        {menuItems.map((item) => (
          <TouchableOpacity
            key={item.screen}
            style={styles.menuButton}
            onPress={() => navigation.navigate(item.screen)}
          >
            <Text style={styles.menuButtonText}>{item.label}</Text>
          </TouchableOpacity>
        ))}
      </View>

      <Text style={styles.sectionTitle}>Potential Scam Alerts</Text>

      <FlatList
        data={scamAlerts}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.alertCard}
            onPress={() => navigation.navigate("AlertDetails", { alert: item })}
          >
            <Text style={styles.alertName}>{item.resident}</Text>
            <Text style={styles.alertType}>{item.type}</Text>
            <Text style={styles.alertRisk}>Risk: {item.risk}</Text>
          </TouchableOpacity>
        )}
        scrollEnabled={false}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: colors.white,
    paddingTop: 60,
    paddingHorizontal: 20,
    paddingBottom: 20,
  },
  header: {
    fontSize: 28,
    fontWeight: "bold",
    color: colors.textDark,
  },
  subHeader: {
    fontSize: 18,
    color: "#666",
    marginBottom: 20,
  },
  menuGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    gap: 12,
    marginBottom: 30,
    justifyContent: "space-between",
  },
  menuButton: {
    flex: 1,
    backgroundColor: colors.pink,
    paddingVertical: 16,
    paddingHorizontal: 12,
    borderRadius: 12,
    minWidth: "48%",
    alignItems: "center",
    justifyContent: "center",
  },
  menuButtonText: {
    fontSize: 14,
    fontWeight: "bold",
    color: colors.textDark,
    textAlign: "center",
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: "bold",
    marginBottom: 15,
  },
  alertCard: {
    backgroundColor: colors.alertCard,
    padding: 16,
    borderRadius: 14,
    marginBottom: 14,
  },
  alertName: {
    fontSize: 18,
    fontWeight: "bold",
    color: colors.textDark,
  },
  alertType: {
    fontSize: 14,
    color: "#555",
    marginTop: 4,
  },
  alertRisk: {
    fontSize: 14,
    marginTop: 8,
    fontWeight: "bold",
    color: "red",
  },
});
