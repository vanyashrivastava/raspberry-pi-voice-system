import React from "react";
import { View, Text, StyleSheet, TouchableOpacity, FlatList, ScrollView } from "react-native";
import { colors } from "../theme/colors";

// Helper function to dynamically determine risk color
const getRiskColor = (risk) => {
  switch (risk.toLowerCase()) {
    case 'high':
      return colors.alertRed;
    case 'medium':
      return colors.alertOrange;
    case 'low':
      return colors.alertGreen;
    default:
      return colors.textSecondary;
  }
};

// Helper function to get an icon for the menu item
const getMenuIcon = (label) => {
    switch (label.split(" ")[0]) {
        case '📧': return '✉️';
        case '📞': return '📱';
        case '👥': return '🧑‍🤝‍🧑';
        case '⚙️': return '⚙️';
        default: return '➡️';
    }
};

export default function HomeScreen({ navigation }) {
  const nursingHomeName = "Silver Oaks Nursing Home";
  const residentsMonitored = 42;

  const scamAlerts = [
    { id: "1", resident: "Evelyn Carter", risk: "High", type: "Bank phishing text" },
    { id: "2", resident: "Howard Miles", risk: "Medium", type: "Medicare scam call" },
    { id: "3", resident: "Martha Lee", risk: "Low", type: "Junk email (not scam)" },
  ];

  const menuItems = [
    { label: "📧 Email Alerts", screen: "EmailAlerts", icon: '✉️' },
    { label: "📞 Call Monitoring", screen: "CallMonitoring", icon: '📱' },
    { label: "👥 Residents", screen: "ResidentList", icon: '🧑‍🤝‍🧑' },
    { label: "⚙️ Settings", screen: "Settings", icon: '⚙️' },
  ];

  return (
    <ScrollView style={styles.container}>
      {/* HEADER SECTION */}
      <View style={styles.headerContainer}>
        <Text style={styles.header}>{nursingHomeName}</Text>
        {/* Placeholder for a small Piggy Logo/Icon */}
        <Text style={styles.logoIcon}>🐷</Text> 
      </View>

      {/* MONITORING STATS CARD */}
      <View style={styles.statsCard}>
        <Text style={styles.statsNumber}>{residentsMonitored}</Text>
        <Text style={styles.statsLabel}>Residents Monitored</Text>
      </View>
      
      {/* QUICK MENU GRID */}
      <View style={styles.menuGrid}>
        {menuItems.map((item) => (
          <TouchableOpacity
            key={item.screen}
            style={styles.menuButton}
            onPress={() => navigation.navigate(item.screen)}
          >
            <Text style={styles.menuIcon}>{item.icon}</Text>
            <Text style={styles.menuButtonText}>{item.label.split(" ")[1]}</Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* ALERT SECTION */}
      <Text style={styles.sectionTitle}>⚠️ Potential Scam Alerts</Text>

      <FlatList
        data={scamAlerts}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => (
          <TouchableOpacity
            style={styles.alertCard(getRiskColor(item.risk))}
            onPress={() => navigation.navigate("AlertDetails", { alert: item })}
          >
            <View style={styles.alertHeader}>
              <Text style={styles.alertRiskText(getRiskColor(item.risk))}>
                {item.risk.toUpperCase()}
              </Text>
              <Text style={styles.alertType}>{item.type}</Text>
            </View>
            <Text style={styles.alertName}>{item.resident}</Text>
          </TouchableOpacity>
        )}
        scrollEnabled={false}
      />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: colors.background, // Use a light background color
    paddingTop: 50, // More space at the top
    paddingHorizontal: 20,
  },
  headerContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 10,
  },
  logoIcon: {
      fontSize: 28, // Small, cartoony logo
  },
  header: {
    fontSize: 28,
    fontWeight: "800", // Semi-bold/Extra-bold for modern look
    color: colors.textDark,
  },
  
  // --- Stats Card ---
  statsCard: {
    backgroundColor: colors.lightGreen, // Green for a positive/monitoring status
    padding: 25,
    borderRadius: 20,
    alignItems: 'center',
    marginBottom: 30,
    elevation: 4,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  statsNumber: {
    fontSize: 48,
    fontWeight: '900',
    color: colors.textDark,
  },
  statsLabel: {
    fontSize: 16,
    color: colors.textDark,
    opacity: 0.7,
    marginTop: 5,
  },

  // --- Menu Grid ---
  menuGrid: {
    flexDirection: "row",
    flexWrap: "wrap",
    justifyContent: "space-between",
    marginBottom: 30,
  },
  menuButton: {
    width: "48%", // Ensure two columns
    backgroundColor: colors.pinkLight, // Light pink for navigation
    padding: 20,
    borderRadius: 16,
    marginBottom: 15,
    alignItems: "flex-start", // Icons and text align left
    elevation: 2,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  },
  menuIcon: {
      fontSize: 28,
      marginBottom: 8,
  },
  menuButtonText: {
    fontSize: 16,
    fontWeight: "bold",
    color: colors.textDark,
    textAlign: "left",
  },
  
  // --- Alerts Section ---
  sectionTitle: {
    fontSize: 20,
    fontWeight: "800",
    color: colors.textDark,
    marginBottom: 15,
  },
  // Function to style the alert card border/background
  alertCard: (riskColor) => ({
    backgroundColor: colors.white,
    borderLeftWidth: 6, // Emphasize the risk level with a border
    borderLeftColor: riskColor,
    padding: 16,
    borderRadius: 12,
    marginBottom: 12,
    elevation: 3,
    shadowColor: colors.shadow,
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 3,
  }),
  alertHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 4,
  },
  // Function to style the risk text color
  alertRiskText: (riskColor) => ({
    fontSize: 12,
    fontWeight: "900",
    color: riskColor,
  }),
  alertName: {
    fontSize: 18,
    fontWeight: "800",
    color: colors.textDark,
  },
  alertType: {
    fontSize: 14,
    color: colors.textSecondary,
    fontStyle: 'italic',
  },
});

// --- Suggested `colors` object structure for context ---
/* // In ../theme/colors.js (Based on the Pink/Green theme)
export const colors = {
    background: '#F9F9F9',         // Very light grey/white background
    textDark: '#2C3E50',           // Dark charcoal for primary text
    textSecondary: '#6C7A89',      // Lighter grey for secondary text
    shadow: '#000000',

    // Theme Colors (Pink/Green)
    pinkLight: '#FFDCEF',          // Light pink for menu cards (soft, cartoony)
    lightGreen: '#A0E4B0',         // Soft green for the stats card

    // Alert Colors (Professional/Safety-focused)
    alertRed: '#E74C3C',           // Bold Red for High Risk
    alertOrange: '#F39C12',        // Orange for Medium Risk
    alertGreen: '#2ECC71',         // Green for Low/Safe status (or use lightGreen here)
    white: '#FFFFFF',
};
*/